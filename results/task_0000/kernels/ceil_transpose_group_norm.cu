#include <cuda_runtime.h>
#include <stdint.h>

struct Pair {
    float sum;
    float sumsq;
};

__device__ __forceinline__ Pair warp_reduce(Pair v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.sum += __shfl_down_sync(0xffffffffu, v.sum, offset);
        v.sumsq += __shfl_down_sync(0xffffffffu, v.sumsq, offset);
    }
    return v;
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK) void ceil_transpose_group_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int height,
    int width,
    int groups) {
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int channels_per_group = height / groups;
    const int elems = channels_per_group * channels * width;
    const int h0 = group * channels_per_group;

    Pair local = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < elems; i += BLOCK) {
        const int w = i % width;
        const int t = i / width;
        const int c = t % channels;
        const int h = h0 + t / channels;
        const int64_t input_idx =
            ((int64_t(batch) * channels + c) * height + h) * width + w;
        const float v = ceilf(input[input_idx]);
        local.sum += v;
        local.sumsq = fmaf(v, v, local.sumsq);
    }

    local = warp_reduce(local);
    __shared__ Pair warp_sums[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = local;
    __syncthreads();

    Pair total = {0.0f, 0.0f};
    if (warp == 0) {
        if (lane < BLOCK / 32) total = warp_sums[lane];
        total = warp_reduce(total);
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();

    const float inv_count = 1.0f / float(elems);
    const float mean = warp_sums[0].sum * inv_count;
    float variance = warp_sums[0].sumsq * inv_count - mean * mean;
    variance = fmaxf(variance, 0.0f);
    const float inv_std = rsqrtf(variance + 1.0e-5f);

    const int64_t output_base =
        (int64_t(batch) * height + h0) * channels * width;
    for (int i = threadIdx.x; i < elems; i += BLOCK) {
        const int w = i % width;
        const int t = i / width;
        const int c = t % channels;
        const int h = h0 + t / channels;
        const int64_t input_idx =
            ((int64_t(batch) * channels + c) * height + h) * width + w;
        const float v = ceilf(input[input_idx]);
        output[output_base + i] = (v - mean) * inv_std;
    }
}

extern "C" void ceil_transpose_group_norm_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    int groups,
    cudaStream_t stream) {
    dim3 grid(groups, batch);
    ceil_transpose_group_norm_kernel<256><<<grid, 256, 0, stream>>>(
        input, output, channels, height, width, groups);
}
