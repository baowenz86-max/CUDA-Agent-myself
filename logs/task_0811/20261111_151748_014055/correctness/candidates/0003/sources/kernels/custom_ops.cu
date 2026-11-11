#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// Bilinear upsampling (align_corners=false)
// -----------------------------------------------------------------------------
// Input: 4D tensor with shape [N, C, H, W], arbitrary strides.
// Output: 4D tensor with shape [N, C, outH, outW], contiguous.
// scale_factor is a scalar float (spatial dims multiplied by it).
// outH = floor(H * scale_factor), outW = floor(W * scale_factor).
// For reference nn.Upsample(scale_factor=2.0, mode='bilinear') with default align_corners=False.
// Geometry calculations follow PyTorch's upsample_bilinear2d behavior.

__global__ void upsample_bilinear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float rheight, float rwidth,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= outW || out_y >= outH) return;

    float h1r = rheight * (out_y + 0.5f) - 0.5f;
    h1r = h1r < 0 ? 0 : h1r;
    int h1 = (int)h1r;
    int h1p = (h1 < H - 1) ? 1 : 0;
    float h1lambda = h1r - h1;
    float h0lambda = 1.0f - h1lambda;

    float w1r = rwidth * (out_x + 0.5f) - 0.5f;
    w1r = w1r < 0 ? 0 : w1r;
    int w1 = (int)w1r;
    int w1p = (w1 < W - 1) ? 1 : 0;
    float w1lambda = w1r - w1;
    float w0lambda = 1.0f - w1lambda;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* in_base = input + n * in_stride_n + c * in_stride_c;
            float val = 
                h0lambda * (w0lambda * in_base[h1 * in_stride_h + w1 * in_stride_w] +
                            w1lambda * in_base[h1 * in_stride_h + (w1 + w1p) * in_stride_w]) +
                h1lambda * (w0lambda * in_base[(h1 + h1p) * in_stride_h + w1 * in_stride_w] +
                            w1lambda * in_base[(h1 + h1p) * in_stride_h + (w1 + w1p) * in_stride_w]);
            output[((n * C + c) * outH + out_y) * outW + out_x] = val;
        }
    }
}

extern "C" void custom_upsample_bilinear_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    double scale_factor,
    int64_t outH, int64_t outW,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    float rheight = (float)H / (float)outH;
    float rwidth = (float)W / (float)outW;
    upsample_bilinear_kernel<<<grid, block, 0, stream>>>(
        input, output,
        (int)N, (int)C, (int)H, (int)W,
        (int)outH, (int)outW,
        rheight, rwidth,
        (long long)in_stride_n, (long long)in_stride_c,
        (long long)in_stride_h, (long long)in_stride_w);
}

// -----------------------------------------------------------------------------
// Softmax over an arbitrary dimension
// -----------------------------------------------------------------------------
// Input: tensor of shape [d0, d1, ..., dn-1], arbitrary strides.
// Output: same shape, contiguous.
// dim: dimension to normalize over (0 <= dim < n).
// We use one thread per output element. Each thread computes its multi-dimensional
// indices, then loops over the softmax dimension to compute max, then sum, then writes.
// This avoids any complex stride aggregation and is correct for arbitrary strides.

__global__ void softmax_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_numel,
    int64_t dim_size,
    int64_t dim_stride,  // input stride for softmax dimension
    int64_t inner_size,  // product of dimensions after dim
    const int64_t* __restrict__ shape,   // shape array on device
    const int64_t* __restrict__ stride,  // input strides array on device
    int64_t ndim,
    int64_t dim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_numel) return;

    // Decompose idx into multi-dimensional indices.
    int64_t remaining = idx;
    int64_t indices[8]; // assume ndim <= 8 for this task
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t s = shape[d];
        indices[d] = remaining % s;
        remaining /= s;
    }

    // Compute base offset for the element, excluding the softmax dimension.
    int64_t base_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            base_offset += indices[d] * stride[d];
        }
    }

    // Output index is simply idx (since output is contiguous and same shape).
    float* out_ptr = output + idx;

    // Compute max over softmax dimension.
    float max_val = -FLT_MAX;
    for (int64_t j = 0; j < dim_size; ++j) {
        float v = input[base_offset + j * dim_stride];
        if (v > max_val) max_val = v;
    }

    // Compute sum of exp.
    float sum = 0.0f;
    for (int64_t j = 0; j < dim_size; ++j) {
        sum += __expf(input[base_offset + j * dim_stride] - max_val);
    }

    // Normalize and write the softmax dimension slice for this output element.
    // But note: one thread per output element, so we only write one value.
    // However, the output element at idx has a specific value for its softmax index.
    // We need to know the index along the softmax dimension for this thread.
    // Instead of one thread per element, we can have one thread per (outer,inner) pair
    // and then loop over softmax dimension. Since we already have the indices,
    // we can compute the value for this specific element.
    int64_t softmax_index = indices[dim];
    out_ptr[0] = __expf(input[base_offset + softmax_index * dim_stride] - max_val) * (1.0f / sum);
}

extern "C" void custom_softmax_dim_launcher(
    const float* input,
    float* output,
    const int64_t* shape,
    int64_t ndim,
    int64_t dim,
    const int64_t* stride, // input strides in elements
    int64_t total_numel,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim || total_numel == 0) return;

    int64_t dim_size = shape[dim];
    int64_t dim_stride = stride[dim];

    // Copy shape and stride to device memory for kernel access.
    int64_t* d_shape;
    int64_t* d_stride;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_stride, stride, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int block = 256;
    int grid = (int)((total_numel + block - 1) / block);

    softmax_dim_kernel<<<grid, block, 0, stream>>>(
        input, output, total_numel, dim_size, dim_stride,
        0, // unused inner_size placeholder
        d_shape, d_stride, ndim, dim);

    // Free device memory after kernel completion (assuming stream ordered).
    cudaFree(d_shape);
    cudaFree(d_stride);
}
