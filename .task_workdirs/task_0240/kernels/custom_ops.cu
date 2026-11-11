#include <cuda_runtime.h>
#include <cstdint>

// Elementwise binary cross entropy with logits for float32.
// Formula: max(x,0) - x*y + log1p(exp(-abs(x)))
// Inputs may have arbitrary strides; output has same strides as y_pred (assumed same as y_true per contract).
// Grid-stride loop handles any numel.
__global__ void bce_with_logits_kernel(
    float* __restrict__ output,
    const float* __restrict__ y_pred,
    const float* __restrict__ y_true,
    int64_t numel,
    int64_t pred_stride_0,
    int64_t pred_stride_1,
    int64_t pred_stride_2,
    int64_t pred_stride_3,
    int64_t true_stride_0,
    int64_t true_stride_1,
    int64_t true_stride_2,
    int64_t true_stride_3,
    int64_t out_stride_0,
    int64_t out_stride_1,
    int64_t out_stride_2,
    int64_t out_stride_3)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        // Compute linear index to 4D coordinates with shape [N,1,H,W] (special case for dim1=1)
        // More general: decode using known dimensions? We only know strides and numel.
        // We assume shape is [128,1,1024,1024] per contract, but to be safe we can compute coordinates from linear index assuming dim1=1.
        // But we don't have actual dimensions, only strides. For arbitrary shape with dims [d0,d1,d2,d3], linear index i = ((i0*d1 + i1)*d2 + i2)*d3 + i3.
        // We can decode if we know d1,d2,d3. We only have strides, not dimensions. However we can use the fact that strides give the mapping: i0 = i / stride0? Not generally.
        // Better approach: pass the actual dimensions as additional parameters, or use a generic function that computes pointer from linear index if strides are contiguous? 
        // For correctness, we should pass dimensions or use the fact that the tensor has 4 dims and we can compute coordinates by successive division if we know dims.
        // Contract says shape [128,1,1024,1024], so we can hardcode dimensions? That would break other shapes. We need a general kernel.
        // The launcher can compute dims and pass them. We'll add dim0, dim1, dim2, dim3 parameters.
    }
}

// We'll need dims to decode linear index. Modify kernel signature to include dims.
// Actually we can avoid decoding by using per-dimension loops if we pass dims and strides. But grid-stride on linear index requires decode.
// Let's rewrite kernel with dimensions passed.

__global__ void bce_with_logits_kernel_dims(
    float* __restrict__ output,
    const float* __restrict__ y_pred,
    const float* __restrict__ y_true,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t pred_stride_0,
    int64_t pred_stride_1,
    int64_t pred_stride_2,
    int64_t pred_stride_3,
    int64_t true_stride_0,
    int64_t true_stride_1,
    int64_t true_stride_2,
    int64_t true_stride_3,
    int64_t out_stride_0,
    int64_t out_stride_1,
    int64_t out_stride_2,
    int64_t out_stride_3)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = dim0 * dim1 * dim2 * dim3;
    if (idx >= total) return;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t i0 = i / (dim1 * dim2 * dim3);
        int64_t rem = i % (dim1 * dim2 * dim3);
        int64_t i1 = rem / (dim2 * dim3);
        rem %= (dim2 * dim3);
        int64_t i2 = rem / dim3;
        int64_t i3 = rem % dim3;

        float x = y_pred[i0 * pred_stride_0 + i1 * pred_stride_1 + i2 * pred_stride_2 + i3 * pred_stride_3];
        float y = y_true[i0 * true_stride_0 + i1 * true_stride_1 + i2 * true_stride_2 + i3 * true_stride_3];
        float max_val = fmaxf(x, 0.0f);
        float loss = max_val - x * y + log1pf(expf(-fabsf(x)));
        output[i0 * out_stride_0 + i1 * out_stride_1 + i2 * out_stride_2 + i3 * out_stride_3] = loss;
    }
}

// Host launcher called from binding (extern "C")
extern "C" void bce_with_logits_forward_launcher(
    float* output,
    const float* y_pred,
    const float* y_true,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t pred_stride_0,
    int64_t pred_stride_1,
    int64_t pred_stride_2,
    int64_t pred_stride_3,
    int64_t true_stride_0,
    int64_t true_stride_1,
    int64_t true_stride_2,
    int64_t true_stride_3,
    int64_t out_stride_0,
    int64_t out_stride_1,
    int64_t out_stride_2,
    int64_t out_stride_3,
    cudaStream_t stream)
{
    int64_t total = dim0 * dim1 * dim2 * dim3;
    if (total == 0) return;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Grid cap: 2^31-1 blocks, but we can use a smaller grid and grid-stride loop handles rest.
    // For total up to ~1.3e8, blocks ~524288 which is fine.
    if (blocks > 65535) blocks = 65535; // optional cap
    bce_with_logits_kernel_dims<<<blocks, threads, 0, stream>>>(
        output, y_pred, y_true,
        dim0, dim1, dim2, dim3,
        pred_stride_0, pred_stride_1, pred_stride_2, pred_stride_3,
        true_stride_0, true_stride_1, true_stride_2, true_stride_3,
        out_stride_0, out_stride_1, out_stride_2, out_stride_3);
}
