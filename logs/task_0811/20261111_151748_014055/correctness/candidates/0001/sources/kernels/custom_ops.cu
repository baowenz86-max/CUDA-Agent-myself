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

    // Compute source coordinates using PyTorch's area_pixel_compute_source_index
    // with align_corners=false and cubic=false: src = scale * (dst + 0.5) - 0.5
    // where scale = input_size / output_size
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
// We use a stable algorithm: subtract max along dim, then exp, then divide by sum.
// The kernel is launched with one thread per element in the "outer" part
// (product of dimensions before and after dim), and each thread loops over
// the softmax dimension to compute max, then sum, then writes normalized values.

__global__ void softmax_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,   // product of dimensions before dim
    int dim_size,     // size of the softmax dimension
    int inner_size,   // product of dimensions after dim
    int64_t stride_outer, // stride of the dimension before dim (for indexing)
    int64_t stride_dim,   // stride of the softmax dimension
    int64_t stride_inner) // stride of the next dimension after dim (usually 1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;

    int outer = idx / inner_size;
    int inner = idx - outer * inner_size;

    const float* data = input + outer * stride_outer + inner * stride_inner;
    float* out_data = output + outer * dim_size * inner_size + inner;

    // max over dim
    float max_val = -FLT_MAX;
    for (int d = 0; d < dim_size; ++d) {
        float v = data[d * stride_dim];
        if (v > max_val) max_val = v;
    }

    // sum of exp
    float sum = 0.0f;
    for (int d = 0; d < dim_size; ++d) {
        sum += __expf(data[d * stride_dim] - max_val);
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (int d = 0; d < dim_size; ++d) {
        out_data[d * inner_size] = __expf(data[d * stride_dim] - max_val) * inv_sum;
    }
}

// Helper to compute strides from shape (contiguous input assumed for output)
// For arbitrary input strides, we pass the actual strides and the dim index.
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
    // Handle negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return; // should not happen, binding validates

    int64_t dim_size = shape[dim];
    if (dim_size == 0) return;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t stride_outer = (dim > 0) ? stride[dim - 1] : 0;
    int64_t stride_dim = stride[dim];
    int64_t stride_inner = (dim + 1 < ndim) ? stride[dim + 1] : 1;

    // For contiguous output we need to compute positions differently.
    // Here we assume output is contiguous with shape given by shape,
    // so we need to map output indices accordingly.
    // Our kernel writes to out_data[d * inner_size], which assumes
    // contiguous layout for the dimensions after dim; but we need to
    // account for outer stride as well. However, outer loop iterates over
    // outer_size * inner_size combinations, and out_data is set to
    // output + outer * dim_size * inner_size + inner, which is correct
    // for contiguous output (outer dimensions have stride = dim_size*inner_size,
    // then inner has stride 1). This works regardless of input strides.

    int total = (int)(outer_size * inner_size);
    int block = 256;
    int grid = (total + block - 1) / block;

    softmax_dim_kernel<<<grid, block, 0, stream>>>(
        input, output,
        (int)outer_size, (int)dim_size, (int)inner_size,
        stride_outer, stride_dim, stride_inner);
}
