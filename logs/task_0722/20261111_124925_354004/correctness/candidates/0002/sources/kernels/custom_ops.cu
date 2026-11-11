#include <cuda_runtime.h>
#include <cstdint>

__host__ __device__ __forceinline__ int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <typename scalar_t>
__global__ void custom_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride_N, int64_t in_stride_C, int64_t in_stride_H, int64_t in_stride_W,
    int64_t out_stride_N, int64_t out_stride_C, int64_t out_stride_W
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_outputs = N * C * W;
    if (idx >= total_outputs) return;

    int64_t w = idx % W;
    int64_t tmp = idx / W;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    const scalar_t* base = input + n * in_stride_N + c * in_stride_C + w * in_stride_W;
    float sum = 0.0f;
    for (int64_t h = 0; h < H; ++h) {
        sum += static_cast<float>(base[h * in_stride_H]);
    }
    output[n * out_stride_N + c * out_stride_C + w * out_stride_W] = static_cast<scalar_t>(sum / H);
}

extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride_N, int64_t in_stride_C, int64_t in_stride_H, int64_t in_stride_W,
    int64_t out_stride_N, int64_t out_stride_C, int64_t out_stride_W,
    cudaStream_t stream
) {
    int64_t total_outputs = N * C * W;
    int block = 256;
    int64_t grid = div_up(total_outputs, block);
    custom_mean_kernel<float><<<grid, block, 0, stream>>>(
        input, output,
        N, C, H, W,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W,
        out_stride_N, out_stride_C, out_stride_W
    );
}

template <typename scalar_t>
__global__ void custom_gt_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ other,
    bool* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride_N, int64_t in_stride_C, int64_t in_stride_H, int64_t in_stride_W,
    int64_t other_stride_N, int64_t other_stride_C, int64_t other_stride_W,
    int64_t out_stride_N, int64_t out_stride_C, int64_t out_stride_H, int64_t out_stride_W
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = N * C * H * W;
    if (idx >= total_elements) return;

    int64_t w = idx % W;
    int64_t tmp = idx / W;
    int64_t h = tmp % H;
    tmp = tmp / H;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    scalar_t input_val = input[n * in_stride_N + c * in_stride_C + h * in_stride_H + w * in_stride_W];
    scalar_t other_val = other[n * other_stride_N + c * other_stride_C + w * other_stride_W];
    output[n * out_stride_N + c * out_stride_C + h * out_stride_H + w * out_stride_W] =
        (input_val > other_val);
}

extern "C" void custom_gt_launcher(
    const float* input,
    const float* other,
    bool* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride_N, int64_t in_stride_C, int64_t in_stride_H, int64_t in_stride_W,
    int64_t other_stride_N, int64_t other_stride_C, int64_t other_stride_W,
    int64_t out_stride_N, int64_t out_stride_C, int64_t out_stride_H, int64_t out_stride_W,
    cudaStream_t stream
) {
    int64_t total_elements = N * C * H * W;
    int block = 256;
    int64_t grid = div_up(total_elements, block);
    custom_gt_kernel<float><<<grid, block, 0, stream>>>(
        input, other, output,
        N, C, H, W,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W,
        other_stride_N, other_stride_C, other_stride_W,
        out_stride_N, out_stride_C, out_stride_H, out_stride_W
    );
}
