#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void ceil_transpose_group_norm_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int channels, int height, int width, int group_height,
    int vectors_per_group) {
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int h_begin = group * group_height;
    const int input_batch_stride = channels * height * width;
    const int output_batch_stride = height * channels * width;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int vector_index = threadIdx.x; vector_index < vectors_per_group;
         vector_index += THREADS) {
        const int scalar_index = vector_index * 4;
        const int h_local = scalar_index / (channels * width);
        const int remainder = scalar_index - h_local * channels * width;
        const int channel = remainder / width;
        const int w = remainder - channel * width;
        const float4 values = *reinterpret_cast<const float4*>(
            input + batch * input_batch_stride + channel * height * width
            + (h_begin + h_local) * width + w);
        const float x0 = ceilf(values.x);
        const float x1 = ceilf(values.y);
        const float x2 = ceilf(values.z);
        const float x3 = ceilf(values.w);
        sum += (x0 + x1) + (x2 + x3);
        square_sum += (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3);
    }

    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    __shared__ float warp_sums[THREADS / 32];
    __shared__ float warp_squares[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_squares[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < THREADS / 32 ? warp_sums[lane] : 0.0f;
        square_sum = lane < THREADS / 32 ? warp_squares[lane] : 0.0f;
        sum = warp_sum(sum);
        square_sum = warp_sum(square_sum);
        if (lane == 0) {
            const float count = float(vectors_per_group * 4);
            const float mean = sum / count;
            const float variance = fmaxf(square_sum / count - mean * mean, 0.0f);
            warp_sums[0] = mean;
            warp_squares[0] = rsqrtf(variance + 1.0e-5f);
        }
    }
    __syncthreads();
    const float mean = warp_sums[0];
    const float inverse_std = warp_squares[0];

    for (int vector_index = threadIdx.x; vector_index < vectors_per_group;
         vector_index += THREADS) {
        const int scalar_index = vector_index * 4;
        const int h_local = scalar_index / (channels * width);
        const int remainder = scalar_index - h_local * channels * width;
        const int channel = remainder / width;
        const int w = remainder - channel * width;
        const float4 values = *reinterpret_cast<const float4*>(
            input + batch * input_batch_stride + channel * height * width
            + (h_begin + h_local) * width + w);
        float4 result;
        result.x = (ceilf(values.x) - mean) * inverse_std;
        result.y = (ceilf(values.y) - mean) * inverse_std;
        result.z = (ceilf(values.z) - mean) * inverse_std;
        result.w = (ceilf(values.w) - mean) * inverse_std;
        *reinterpret_cast<float4*>(
            output + batch * output_batch_stride
            + (h_begin + h_local) * channels * width + channel * width + w) = result;
    }
}

extern "C" void ceil_transpose_group_norm_launcher(
    const float* input, float* output, int batch, int channels, int height,
    int width, int groups, cudaStream_t stream) {
    constexpr int threads = 256;
    const int group_height = height / groups;
    const int vectors_per_group = group_height * channels * width / 4;
    const dim3 grid(groups, batch);
    ceil_transpose_group_norm_kernel<threads><<<grid, threads, 0, stream>>>(
        input, output, channels, height, width, group_height, vectors_per_group);
}
