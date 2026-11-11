#include <cuda_runtime.h>
#include <cstdint>

// custom_abs kernel: element-wise absolute value
template <typename T>
__global__ void custom_abs_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);  // only float version is used
    }
}

// adaptive_avg_pool2d kernel: each thread computes one output element
// input is assumed contiguous float tensor with shape [N, C, H, W]
// output shape [N, C, out_h, out_w]
__global__ void adaptive_avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_out = N * C * out_h * out_w;
    if (idx >= total_out) return;

    // decompose idx into n, c, oh, ow
    int64_t ow = idx % out_w;
    int64_t tmp = idx / out_w;
    int64_t oh = tmp % out_h;
    tmp /= out_h;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    // compute start and end indices using floor/ceil formula
    int64_t h_start = (oh * H) / out_h;
    int64_t h_end = ((oh + 1) * H + out_h - 1) / out_h;  // ceil division
    int64_t w_start = (ow * W) / out_w;
    int64_t w_end = ((ow + 1) * W + out_w - 1) / out_w;

    // ensure valid range
    if (h_start < 0) h_start = 0;
    if (h_end > H) h_end = H;
    if (w_start < 0) w_start = 0;
    if (w_end > W) w_end = W;

    float sum = 0.0f;
    int64_t count = 0;
    const float* base = input + (n * C + c) * H * W;
    for (int64_t h = h_start; h < h_end; ++h) {
        const float* row = base + h * W;
        for (int64_t w = w_start; w < w_end; ++w) {
            sum += row[w];
            ++count;
        }
    }

    // if count == 0 (should not happen), avoid division by zero
    output[idx] = (count > 0) ? (sum / count) : 0.0f;
}

// launcher for custom_abs
extern "C" void custom_abs_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    // cap grid to 2^31-1 blocks (GPU limit) and use grid-stride loop in kernel
    if (blocks > 2147483647) blocks = 2147483647;
    custom_abs_kernel<float><<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(
        input, output, numel);
}

// launcher for adaptive_avg_pool2d
extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    cudaStream_t stream)
{
    int64_t total_out = N * C * out_h * out_w;
    if (total_out == 0) return;
    const int block_size = 256;
    int64_t blocks = (total_out + block_size - 1) / block_size;
    if (blocks > 2147483647) blocks = 2147483647;
    adaptive_avg_pool2d_kernel<<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(
        input, output, N, C, H, W, out_h, out_w);
}
