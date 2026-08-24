#include <cuda_runtime.h>
#include <stdint.h>

// One block owns one (batch, group) row.  The input is read coalesced in
// width-sized runs despite the virtual transpose, and no intermediate tensor
// is materialized.  GroupNorm has no affine parameters in the source model.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void fused_group_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int transposed_channels,
    int width,
    int groups,
    int elements_per_group) {
    const int group_id = blockIdx.x;
    const int n = group_id / groups;
    const int group = group_id - n * groups;

    double sum = 0.0;
    double square_sum = 0.0;
    const int channels_per_group = transposed_channels / groups;
    const int first_h = group * channels_per_group;

    for (int i = threadIdx.x; i < elements_per_group; i += BLOCK) {
        const int w = i % width;
        const int channel = (i / width) % channels;
        const int h = first_h + i / (channels * width);
        const int64_t input_idx =
            ((int64_t(n) * channels + channel) * transposed_channels + h)
            * width + w;
        const float value = ceilf(input[input_idx]);
        sum += double(value);
        square_sum += double(value) * double(value);
    }

    __shared__ double sums[BLOCK / 32];
    __shared__ double squares[BLOCK / 32];
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
        square_sum += __shfl_down_sync(mask, square_sum, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sums[warp] = sum;
        squares[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < BLOCK / 32 ? sums[lane] : 0.0;
        square_sum = lane < BLOCK / 32 ? squares[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset);
            square_sum += __shfl_down_sync(mask, square_sum, offset);
        }
        if (lane == 0) {
            const double inv_count = 1.0 / double(elements_per_group);
            const double mean = sum * inv_count;
            const double variance = fmax(square_sum * inv_count - mean * mean, 0.0);
            sums[0] = mean;
            squares[0] = rsqrt(variance + 1.0e-5);
        }
    }
    __syncthreads();

    const float mean = float(sums[0]);
    const float inv_std = float(squares[0]);
    const int64_t output_base =
        (int64_t(n) * transposed_channels + first_h) * channels * width;
    for (int i = threadIdx.x; i < elements_per_group; i += BLOCK) {
        const int w = i % width;
        const int channel = (i / width) % channels;
        const int h = first_h + i / (channels * width);
        const int64_t input_idx =
            ((int64_t(n) * channels + channel) * transposed_channels + h)
            * width + w;
        const float value = ceilf(input[input_idx]);
        output[output_base + i] = (value - mean) * inv_std;
    }
}

extern "C" void ceil_transpose_group_norm_launcher(
    const float* input, float* output, int batch, int channels,
    int height, int width, int groups, cudaStream_t stream) {
    constexpr int block = 256;
    const int group_size = height / groups;
    const int elements_per_group = group_size * channels * width;
    const int grid = batch * groups;
    fused_group_norm_kernel<block><<<grid, block, 0, stream>>>(
        input, output, channels, height, width, groups,
        elements_per_group);
}
