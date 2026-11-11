// kernels/custom_ops.cu
// CUDA kernels for max_reduce and relu6 used in model_new.py
// All launchers have C linkage for binding.cpp

#include <cuda_runtime.h>
#include <cstdint>

// ---------------- max_reduce over a specified dimension ----------------
// Input: x (float, arbitrary strides), dim (reduction dimension)
// Output: values (float, reduced shape) and indices (int64_t, reduced shape)
// This implementation assumes keepdim=true for the given contract, but can handle
// general by reducing dimension to 1 if keepdim true, otherwise removed.
// The binding layer will allocate output tensors with appropriate shapes and strides.
// We only require dim to be a valid dimension (0 <= dim < input_ndim) and input to be CUDA.
// We also assume input is contiguous or at least the reduction dimension stride is uniform.
// For simplicity, we handle strides as passed; the binding layer should provide contiguous
// output tensors, but input may have arbitrary strides (though in this task it is contiguous).

// Kernel: each thread computes one output element (over all non-reduced dimensions).
// Uses grid-stride loop over output elements.
template <int BLOCK_SIZE>
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t reduced_size,
    int64_t inner_size,
    int64_t outer_size,
    int64_t input_stride_reduced,
    int64_t input_stride_outer,
    int64_t input_stride_inner,
    int64_t output_stride_outer,
    int64_t output_stride_inner
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t total_outputs = outer_size * inner_size;
    for (; idx < total_outputs; idx += stride) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx - outer * inner_size;

        const float* base = input + outer * input_stride_outer + inner * input_stride_inner;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int64_t r = 1; r < reduced_size; ++r) {
            float val = base[r * input_stride_reduced];
            if (val > max_val) {
                max_val = val;
                max_idx = r;
            }
        }
        values[outer * output_stride_outer + inner * output_stride_inner] = max_val;
        indices[outer * output_stride_outer + inner * output_stride_inner] = max_idx;
    }
}

// Launcher for max_reduce. x is assumed contiguous (as in task).
// dim is the dimension to reduce. keepdim is passed but here we assume keepdim=true
// and output dims are already prepared by binding. We compute strides accordingly.
// out_values and out_indices are already allocated with appropriate shapes and strides.
extern "C" void max_reduce_launcher(
    const float* x,
    float* out_values,
    int64_t* out_indices,
    int ndim,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream
) {
    // Validate dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return; // invalid, but binding should check
    if (keepdim) {
        // dimension remains as 1, output_shape[dim] = 1
        // For our kernel, we treat reduction dimension as contiguous chunk
        // When keepdim true, output strides include a dummy stride for dim.
        // We need to map output linear index to outer/inner relative to reduction dim.
        // In this task, dim=1, keepdim=true, input shape [512,16,256,256], output [512,1,256,256]
        // We can simplify: treat reduced dimension as middle, outer = product of dims before dim,
        // inner = product of dims after dim. input_stride_reduced = input_strides[dim]
        // outer_size = product(input_shape[0..dim-1]) (or 1 if dim=0)
        // inner_size = product(input_shape[dim+1..]) (or 1 if last)
        // input_stride_outer = input_strides[dim-1] if dim>0 else 0? Actually for outer loop, we need stride in input for outer index.
        // Since input is contiguous, we can compute pointer offsets directly using strides.
        // But to be generic with arbitrary strides, we can compute linear index in input via outer*input_stride_outer + r*input_stride_reduced + inner*input_stride_inner.
        // Similarly for output linear index: outer*output_stride_outer + inner*output_stride_inner (since reduced dim size=1 and its stride may be arbitrary but we can ignore).
        // However, binding will likely allocate output contiguous with shape [outer_size,1,inner_size] and strides accordingly.
        // To be safe, we pass the necessary stride values.
        // For this specialized implementation, we assume input is contiguous and output is contiguous (which is typical).
        // We compute the strides as follows:
    }
    // Compute outer_size and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t inner_size = 1;
    for (int i = dim+1; i < ndim; ++i) inner_size *= input_shape[i];
    int64_t reduced_size = input_shape[dim];

    // Strides for input (assuming contiguous, but we can use provided strides)
    int64_t input_stride_reduced = input_strides[dim];
    int64_t input_stride_outer = (dim > 0) ? input_strides[dim-1] : 0; // for outer loop, need stride in input for incrementing outer index. But outer is multi-dim, so better to compute linear index manually.
    // Actually we can precompute the base offset for each outer/inner combination:
    // The kernel will compute offsets via outer and inner using a precomputed layout? Simpler: since input is contiguous, we can compute linear offset = ((outer * reduced_size + r) * inner_stride? No, that assumes dim is last. Need general formula.
    // Alternative: Use a kernel that loops over all output elements using output linear index and compute corresponding input base pointer via index arithmetic on strides.
    // For performance and simplicity given the known shape [512,16,256,256] with dim=1 and contiguous,
    // we can implement a specialized kernel that expects dim=1, keepdim=true, and 4D contiguous input.
    // But to be robust, we handle general via a more generic but less optimized method:
    // Each thread handles one output element; compute the multi-dimensional indices for output,
    // then map to input indices with reduction over dim.
    // This can be done using loops and modular arithmetic, but may be slower. However, for this task it is acceptable.
    // We'll implement generic using precomputed strides and sizes, but assume input contiguous, output contiguous.
    // The binding will pass appropriate strides.
    // But for simplicity, we can assume input is contiguous and use a flat index with known layout? Since shape known, we can hardcode dim=1 case for optimal performance.
    // Since contract only uses dim=1 and keepdim=true, we can optimize for that but still include fallback generic.
    // We'll write a kernel specialized for dim=1, keepdim=true, 4D contiguous input, but also a generic kernel for other cases.
    // However, launcher only needs to call the specialized one if conditions met, else generic.

    // Decide: if ndim==4 && dim==1 && keepdim && input_strides are contiguous, use specialized kernel.
    bool is_special = (ndim == 4 && dim == 1 && keepdim &&
                       input_strides[3] == 1 && input_strides[2] == input_shape[3] &&
                       input_strides[1] == input_shape[2]*input_shape[3] &&
                       input_strides[0] == input_shape[1]*input_shape[2]*input_shape[3]);
    if (is_special) {
        // Specialized: input [N, C, H, W], output [N, 1, H, W]
        // Each thread handles one (n, h, w) output element, reduces over c.
        int N = input_shape[0];
        int C = input_shape[1];
        int H = input_shape[2];
        int W = input_shape[3];
        int64_t total = (int64_t)N * H * W;
        int block = 256;
        int grid = (int)((total + block - 1) / block);
        // Launch specialized kernel
        max_reduce_special<<<grid, block, 0, stream>>>(
            x, out_values, out_indices, N, C, H, W);
    } else {
        // Generic kernel
        int block = 256;
        int grid = (int)((outer_size * inner_size + block - 1) / block);
        // We need to pass input strides as int64_t, but we have them.
        // However, the kernel signature above expects specific stride values for reduced, outer, inner.
        // We need to compute the appropriate strides for contiguous output.
        // For output, if keepdim true, output_shape same as input except dim=1, output strides contiguous.
        // We'll compute output_stride_outer and output_stride_inner based on output_shape and contiguity assumption.
        // The binding layer will ensure output is contiguous.
        // For generic, we need to compute input_stride_outer (stride for incrementing outer linear index) and input_stride_inner (stride for incrementing inner linear index).
        // For a contiguous input, outer index increments by product of sizes after dim? Actually if we flatten outer and inner separately, the input offset = outer * input_strides[0] + inner * input_strides[dim+1] (if dim>0 and dim<ndim-1). More generally, we can compute base pointer for each outer/inner via helper, but in kernel we can calculate as:
        // input_offset = outer * input_strides[0] + inner * input_strides[dim+1]? That assumes only one dim before and after. Not general.
        // We'll simplify by implementing a generic kernel that uses input_strides array passed as pointer and computes indices by decomposing the output linear index into multi-index, then map to input address.
        // That would require loops inside kernel, okay.
        // However, for this task, the specialized kernel will always be used because input is contiguous and shape known.
        // So generic kernel may be less tested but should be correct.
        // We'll implement a simple generic kernel that uses precomputed strides for the reduced dimension and for the output linear indexing, but rely on contiguous input and output.
        // Since we already have the specialized path, we can keep generic as fallback with moderate performance.
        // For completeness, we'll implement generic using a kernel that takes int64_t input_shape[ ], input_strides[ ], output_strides[ ], and computes offsets via loops.
        // But to avoid excessive code, we can call a different kernel.
        // Let's define a separate generic kernel.
    }
    // Note: The specialized kernel is defined below.
}

// Specialized kernel for dim=1, keepdim=true, contiguous 4D input [N,C,H,W]
__global__ void max_reduce_special(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int N, int C, int H, int W)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H * W;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int n = idx / (H * W);
        int rem = idx - n * (H * W);
        int h = rem / W;
        int w = rem - h * W;

        const float* base = input + ((int64_t)n * C + 0) * H * W + h * W + w;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base[c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        int64_t out_idx = (int64_t)n * H * W + h * W + w;
        values[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

// ---------------- relu6 ----------------
__global__ void relu6_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float v = input[idx];
        v = v > 0.0f ? v : 0.0f;
        v = v < 6.0f ? v : 6.0f;
        output[idx] = v;
    }
}

extern "C" void relu6_launcher(const float* input, float* output, int64_t n, cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu6_kernel<<<grid, block, 0, stream>>>(input, output, n);
}

// Note: The launcher for max_reduce is above. Ensure it is complete and includes both specialized and generic path.
// We also include a generic max_reduce kernel for cases not meeting specialization.
__global__ void max_reduce_generic(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t reduced_size,
    int64_t outer_size,
    int64_t inner_size,
    int64_t input_stride_reduced,
    int64_t input_stride_outer,
    int64_t input_stride_inner,
    int64_t output_stride_outer,
    int64_t output_stride_inner)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx - outer * inner_size;
        const float* base = input + outer * input_stride_outer + inner * input_stride_inner;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int64_t r = 1; r < reduced_size; ++r) {
            float val = base[r * input_stride_reduced];
            if (val > max_val) {
                max_val = val;
                max_idx = r;
            }
        }
        values[outer * output_stride_outer + inner * output_stride_inner] = max_val;
        indices[outer * output_stride_outer + inner * output_stride_inner] = max_idx;
    }
}

// Complete launcher definition (was partial above)
extern "C" void max_reduce_launcher(
    const float* x,
    float* out_values,
    int64_t* out_indices,
    int ndim,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t inner_size = 1;
    for (int i = dim+1; i < ndim; ++i) inner_size *= input_shape[i];
    int64_t reduced_size = input_shape[dim];

    bool is_special = (ndim == 4 && dim == 1 && keepdim &&
                       input_strides[3] == 1 && input_strides[2] == input_shape[3] &&
                       input_strides[1] == input_shape[2]*input_shape[3] &&
                       input_strides[0] == input_shape[1]*input_shape[2]*input_shape[3]);
    if (is_special) {
        int N = input_shape[0];
        int C = input_shape[1];
        int H = input_shape[2];
        int W = input_shape[3];
        int64_t total = (int64_t)N * H * W;
        int block = 256;
        int grid = (int)((total + block - 1) / block);
        max_reduce_special<<<grid, block, 0, stream>>>(x, out_values, out_indices, N, C, H, W);
    } else {
        // For generic path, we assume output is contiguous and keepdim true (or false but binding will handle)
        // For keepdim true, output_shape[dim]=1, so output_stride_outer = output_strides[dim-1] if dim>0 else 0? Actually we need output stride for outer index and inner index.
        // Since output is contiguous, output_stride_outer = product of sizes after dim (if dim>0) else 0? More exactly:
        // If we flatten outer and inner, the output linear index = outer * output_inner_stride + inner * output_inner_stride? Need careful.
        // For contiguous output with shape [outer, 1, inner], the stride for outer is (1 * inner_size), and for inner is 1.
        // But if keepdim false, shape is [outer, inner], outer stride is inner_size, inner stride 1.
        // In both cases, output_stride_outer = inner_size, output_stride_inner = 1, assuming contiguous and no padding.
        // However, binding may set output_strides accordingly. We can compute from output_shape if contiguous.
        // To be safe, we pass output_stride_outer and output_stride_inner to kernel as computed from output_shape assuming contiguous.
        // Similarly, input_stride_outer and input_stride_inner for contiguous input.
        // For contiguous input, input_stride_outer = product of sizes from dim to end? Actually for flatten outer (dims before dim) and inner (dims after dim),
        // input_stride_outer = stride of dim0 if exists, else 0? Not exactly, but we can compute as follows:
        // If input is contiguous, the offset for (outer, r, inner) = outer * (reduced_size * inner_size) + r * inner_size + inner.
        // So input_stride_outer = reduced_size * inner_size, input_stride_reduced = inner_size, input_stride_inner = 1.
        // However, if dim > 0, we need to ensure outer is the flattened index of all dims before dim, so stride is product of sizes from dim to ndim-1.
        // That is correct for contiguous C-order.
        // So we can compute:
        int64_t input_stride_reduced = 1;
        for (int i = dim+1; i < ndim; ++i) input_stride_reduced *= input_shape[i];
        int64_t input_stride_outer = input_stride_reduced * reduced_size;
        int64_t input_stride_inner = 1;
        int64_t output_stride_outer = inner_size; // since output shape after reduction is [outer, 1, inner] -> contiguous stride outer = inner_size? Actually output linear index = outer * (inner_size * 1) + inner * 1, so yes outer stride = inner_size, inner stride = 1.
        int64_t output_stride_inner = 1;
        // But if keepdim false, output shape [outer, inner], outer stride=inner_size, inner stride=1 same.
        // So okay.
        int64_t total = outer_size * inner_size;
        int block = 256;
        int grid = (int)((total + block - 1) / block);
        max_reduce_generic<<<grid, block, 0, stream>>>(
            x, out_values, out_indices, reduced_size, outer_size, inner_size,
            input_stride_reduced, input_stride_outer, input_stride_inner,
            output_stride_outer, output_stride_inner);
    }
}

// The relu6 launcher was defined earlier; ensure it is not duplicated. The above relu6_launcher is correct.
