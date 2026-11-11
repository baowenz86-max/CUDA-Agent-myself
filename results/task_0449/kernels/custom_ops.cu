#include <cuda_runtime.h>
#include <cmath>

// LRN across channels for 4D tensor [N,C,H,W]
// Each thread computes one output element.
__global__ void lrn_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int half_size = (size - 1) / 2;
    int ceil_half = size / 2;  // for even size, but reference uses floor/ceil as per PyTorch? PyTorch uses: start = max(0, c - (size-1)/2), end = min(C-1, c + size/2)
    // Actually from PyTorch source: int start = c - (size - 1)/2; int end = c + size/2; start = max(0, start); end = min(C-1, end);

    for (; idx < total_elements; idx += stride) {
        // decode n,c,h,w from linear index assuming contiguous row-major: idx = ((n*C + c)*H + h)*W + w
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        // Compute channel window boundaries
        int start = c - half_size;
        int end = c + (size / 2);
        if (start < 0) start = 0;
        if (end >= C) end = C - 1;

        // Sum of squares over channel window at (n, h, w)
        float sum_sq = 0.0f;
        const float* base = input + ((long long)n * C * H * W) + (h * W) + w;
        for (int cc = start; cc <= end; ++cc) {
            float val = base[(long long)cc * H * W];
            sum_sq += val * val;
        }

        float denom = powf(k + alpha * sum_sq, beta);
        float in_val = base[(long long)c * H * W];
        output[idx] = in_val / denom;
    }
}

// Launcher for LRN forward. All tensors are assumed to be contiguous; binding ensures contiguity or we use strides? Contract says supports strides (contiguous or otherwise). Since binding may pass non-contiguous, we should handle strides. But for simplicity and because task input is contiguous, we can require contiguity in binding and call .contiguous() on input if needed. However, launcher only gets raw pointer, cannot see strides. We can add stride parameters for each dimension to support non-contiguous input. But contract says launcher signature should be robust. We'll include stride parameters as int64_t for N,C,H,W dims. The binding will pass strides.

// Updated launcher with strides
__global__ void lrn_kernel_strided(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    long long total_elements,
    int stride_n, int stride_c, int stride_h, int stride_w)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int half_size = (size - 1) / 2;

    for (; idx < total_elements; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        int start = c - half_size;
        int end = c + (size / 2);
        if (start < 0) start = 0;
        if (end >= C) end = C - 1;

        float sum_sq = 0.0f;
        const float* base = input + (long long)n * stride_n + (long long)h * stride_h + (long long)w * stride_w;
        for (int cc = start; cc <= end; ++cc) {
            float val = base[(long long)cc * stride_c];
            sum_sq += val * val;
        }

        float denom = powf(k + alpha * sum_sq, beta);
        float in_val = base[(long long)c * stride_c];
        output[idx] = in_val / denom;
    }
}

// Simple launcher for contiguous input only (strides default to contiguous)
extern "C" void lrn_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    cudaStream_t stream)
{
    // Assume contiguous tensors, so strides: stride_w=1, stride_h=W, stride_c=H*W, stride_n=C*H*W
    int stride_w = 1;
    int stride_h = W;
    int stride_c = H * W;
    int stride_n = C * H * W;

    long long total = (long long)N * C * H * W;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 2147483647) grid = 2147483647; // grid cap; kernel uses grid-stride loop, but if grid capped, still works but may need multiple iterations; we'll use a grid-stride loop with fixed max grid, but here we set grid accordingly and loop handles remainder. Actually if total > block * max_grid, we need grid-stride loop over all. We'll set grid = min(65535, (total+block-1)/block) maybe? Better: use grid-stride loop with grid = min(32000, (total+block-1)/block). Actually we can set grid to cover total, but if total huge and grid > max allowed, kernel launch fails; we cap grid and rely on stride loop. However, the kernel loop is set to iterate with stride = grid*block, so if grid capped below needed, some elements might not be processed if total > grid*block? The loop condition idx < total_elements and idx increments by stride, so if initial idx covers only up to grid*block-1 then increments by stride which is less than total, so all elements will eventually be covered. Yes works.

    int grid_size = (int)((total + block - 1) / block);
    if (grid_size > 2147483647) grid_size = 2147483647;
    if (grid_size < 1) grid_size = 1;

    lrn_kernel_strided<<<grid_size, block, 0, stream>>>(
        output, input, N, C, H, W, size, alpha, beta, k, total,
        stride_n, stride_c, stride_h, stride_w);
}

// Strided version launcher (for non-contiguous)
extern "C" void lrn_forward_strided_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 2147483647) grid = 2147483647;
    if (grid < 1) grid = 1;

    lrn_kernel_strided<<<grid, block, 0, stream>>>(
        output, input, N, C, H, W, size, alpha, beta, k, total,
        stride_n, stride_c, stride_h, stride_w);
}
