// CUDA kernels for max_dim_keepdim and cosine_similarity_dim
#include <cuda_runtime.h>
#include <float.h>

// Helper: compute number of elements before dimension dim for shape and stride
template <typename T>
__device__ T clamp_negative_dim(T dim, T ndim) {
    if (dim < 0) dim += ndim;
    return dim;
}

// Kernel for max reduction along a specified dimension.
// Each block handles one output element (when keepdim=true) or one output element in flattened output space.
// We use a simple approach: gridDim.x = number of outer elements, gridDim.y = number of inner elements,
// blockDim.x = threads to reduce over dim. This assumes dim dimension is contiguous? No, strides may vary.
// To handle arbitrary strides, each thread reads element using computed index.
// For performance, we assume dim is the second dimension of 3D tensor (B,H,W) and keepdim true, but keep general.
// General kernel:
//   output_index = outer * inner_stride_output + inner_element
//   reduce over dim_size elements with stride along dim.
// We launch threads = min(dim_size, 256) or 1024? For large dim_size we need multiple blocks? 
// For simplicity and correctness, use one block per output element, with threads looping with grid-stride over dim.
// But dim_size can be 8192, one block with 1024 threads loops 8 times, that's fine.
// Grid size = outer_size * inner_size. If outer*inner is small (64*128=8192), grid 8192 blocks, okay.
// For arbitrary dim, we compute outer_size = product of dims before dim, inner_size = product after dim, dim_size = shape[dim].
// Output shape: if keepdim, same ndim with dim=1; else ndim-1 with dim removed.
// We'll write kernel that takes pointers, shape, strides for input and output (including indices).
__global__ void max_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ output_values,
                                  int64_t* __restrict__ output_indices,
                                  const int64_t* __restrict__ input_shape,
                                  const int64_t* __restrict__ input_strides,
                                  const int64_t* __restrict__ output_shape,
                                  const int64_t* __restrict__ output_strides,
                                  int ndim,
                                  int dim,
                                  int dim_size,
                                  int64_t outer_size,
                                  int64_t inner_size)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int64_t outer = out_idx / inner_size;
    int64_t inner = out_idx % inner_size;

    // Compute input base offset for this output element.
    int64_t input_base = 0;
    int64_t rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = input_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        input_base += coord * input_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = input_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        input_base += coord * input_strides[i];
    }
    int64_t stride_dim = input_strides[dim];

    float max_val = -FLT_MAX;
    int64_t max_idx = 0;
    for (int64_t i = 0; i < dim_size; ++i) {
        float val = input[input_base + i * stride_dim];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Compute output offset using output_shape and output_strides.
    int64_t output_offset = 0;
    if (output_strides != nullptr) {
        rem = outer;
        for (int i = 0; i < dim; i++) {
            int64_t extent = output_shape[i];
            int64_t coord = rem % extent;
            rem /= extent;
            output_offset += coord * output_strides[i];
        }
        rem = inner;
        for (int i = ndim - 1; i > dim; i--) {
            int64_t extent = output_shape[i];
            int64_t coord = rem % extent;
            rem /= extent;
            output_offset += coord * output_strides[i];
        }
    }
    output_values[output_offset] = max_val;
    output_indices[output_offset] = max_idx;
}

// Launcher for max reduction.
extern "C" void max_dim_keepdim_launcher(
    const float* input,
    float* output_values,
    int64_t* output_indices,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    int64_t dim_size = input_shape[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= input_shape[i];
    int64_t total_out = outer_size * inner_size;

    dim3 grid(total_out);
    dim3 block(1);
    max_reduce_kernel<<<grid, block, 0, stream>>>(
        input, output_values, output_indices,
        input_shape, input_strides, output_shape, output_strides,
        ndim, dim, dim_size, outer_size, inner_size);
}

// --------------------------- Cosine Similarity Kernel ---------------------------
// Compute cosine similarity along dimension dim with broadcasting.
// x1 shape [..., dim_size_at_dim, ...], x2 shape broadcastable with x1.
// Output shape: same as x1 with dim removed.
// We assume x2 has size 1 at dim (as in reference), but general broadcasting is allowed: x2's size at dim is either 1 or equal to x1's size.
// Kernel: each block handles one output element (over dimensions except dim).
// For each output element, iterate over dim_size, compute dot product and norms.
// Formula: dot = sum(x1[i]*x2[i]), norm1 = sum(x1[i]^2), norm2 = sum(x2[i]^2).
// Then output = dot / (sqrt(norm1)*sqrt(norm2) + epsilon) with epsilon=1e-8.
// Need to handle broadcasting: for x2, if x2's dimension at dim is 1, we use same value for all i; if equal, use i.
// General approach: compute index for x2 along dim based on broadcasted dimension.
// We'll assume x2 has same ndim as x1; if not, it can be broadcasted from trailing, but here x2 has shape [B,1,W], so ndim=3.
// We'll compute x2_base and x2_stride_dim similar to x1, but if x2_shape[dim] == 1, then stride along dim may be arbitrary but we treat it as 0 offset for each i? Actually x2 has size 1 at dim, so any element along that dimension has same value; the stride can be anything (likely 0, but not necessarily). We can set effective stride to 0 if size==1.
__global__ void cosine_similarity_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ output,
    const int64_t* __restrict__ x1_shape,
    const int64_t* __restrict__ x1_strides,
    const int64_t* __restrict__ x2_shape,
    const int64_t* __restrict__ x2_strides,
    const int64_t* __restrict__ output_shape,
    const int64_t* __restrict__ output_strides,
    int ndim,
    int dim,
    int dim_size,
    int64_t outer_size,
    int64_t inner_size)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int64_t outer = out_idx / inner_size;
    int64_t inner = out_idx % inner_size;

    // Compute base offsets for x1 and x2 for this output element.
    int64_t x1_base = 0;
    int64_t rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x1_base += coord * x1_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x1_base += coord * x1_strides[i];
    }
    int64_t x1_stride_dim = x1_strides[dim];

    // For x2, we need to compute base similarly, but consider broadcasting.
    // x2 may have different shape; we assume it broadcasts to x1's shape.
    // We compute x2_base using the same outer/inner decomposition, but using x2_shape and x2_strides.
    // If x2 has fewer dimensions, we may need to align; for simplicity, assume ndim same.
    // We'll compute x2_base.
    int64_t x2_base = 0;
    rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = (x2_shape != nullptr) ? x2_shape[i] : x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x2_base += coord * (x2_strides != nullptr ? x2_strides[i] : 0);
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = (x2_shape != nullptr) ? x2_shape[i] : x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x2_base += coord * (x2_strides != nullptr ? x2_strides[i] : 0);
    }
    int64_t x2_stride_dim = (x2_strides != nullptr) ? x2_strides[dim] : 0;
    // If x2's size at dim is 1, we should treat stride as 0 to use same value for all i.
    if (x2_shape != nullptr && x2_shape[dim] == 1) {
        x2_stride_dim = 0;
    }

    float dot = 0.f;
    float norm1 = 0.f;
    float norm2 = 0.f;
    for (int64_t i = 0; i < dim_size; ++i) {
        float v1 = x1[x1_base + i * x1_stride_dim];
        float v2 = x2[x2_base + i * x2_stride_dim];
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    const float eps = 1e-8f;
    float denom = sqrtf(norm1) * sqrtf(norm2) + eps;
    float result = dot / denom;

    // Compute output offset using output_shape and output_strides.
    int64_t output_offset = 0;
    if (output_strides != nullptr) {
        rem = outer;
        for (int i = 0; i < dim; i++) {
            int64_t extent = output_shape[i];
            int64_t coord = rem % extent;
            rem /= extent;
            output_offset += coord * output_strides[i];
        }
        rem = inner;
        for (int i = ndim - 1; i > dim; i--) {
            int64_t extent = output_shape[i];
            int64_t coord = rem % extent;
            rem /= extent;
            output_offset += coord * output_strides[i];
        }
    }
    output[output_offset] = result;
}

// Launcher for cosine similarity.
extern "C" void cosine_similarity_dim_launcher(
    const float* x1,
    const float* x2,
    float* output,
    const int64_t* x1_shape,
    const int64_t* x1_strides,
    const int64_t* x2_shape,
    const int64_t* x2_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    int64_t dim_size = x1_shape[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x1_shape[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= x1_shape[i];
    int64_t total_out = outer_size * inner_size;

    dim3 grid(total_out);
    dim3 block(1);
    cosine_similarity_kernel<<<grid, block, 0, stream>>>(
        x1, x2, output,
        x1_shape, x1_strides, x2_shape, x2_strides, output_shape, output_strides,
        ndim, dim, dim_size, outer_size, inner_size);
}
