#include <cuda_runtime.h>
#include <stdint.h>

// SASS evidence: two pointers and five u32 parameters; maxntid 256;
// CTAID.X selects groups and CTAID.Y selects batches.
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int threads>
__global__ __launch_bounds__(threads, 2)
void ceil_transpose_group_norm_kernel(
    const float* arg0, float* arg1, int arg2, int arg3, int arg4, int arg5,
    int arg6) {
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int group_begin = group * arg5;
    const int input_batch_stride = arg2 * arg3 * arg4;
    const int output_batch_stride = arg3 * arg2 * arg4;
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (int vector_index = threadIdx.x; vector_index < arg6;
         vector_index += threads) {
        const int scalar_index = vector_index * 4;
        const int local_row = scalar_index / (arg2 * arg4);
        const int remainder = scalar_index - local_row * arg2 * arg4;
        const int channel = remainder / arg4;
        const int column = remainder - channel * arg4;
        const float4 values = *reinterpret_cast<const float4*>(
            arg0 + batch * input_batch_stride + channel * arg3 * arg4
            + (group_begin + local_row) * arg4 + column);
        const float value0 = ceilf(values.x);
        const float value1 = ceilf(values.y);
        const float value2 = ceilf(values.z);
        const float value3 = ceilf(values.w);
        sum += (value0 + value1) + (value2 + value3);
        square_sum += (value0 * value0 + value1 * value1) +
                      (value2 * value2 + value3 * value3);
    }

    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    __shared__ float warp_sums[threads / 32];
    __shared__ float warp_squares[threads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_squares[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < threads / 32 ? warp_sums[lane] : 0.0f;
        square_sum = lane < threads / 32 ? warp_squares[lane] : 0.0f;
        sum = warp_sum(sum);
        square_sum = warp_sum(square_sum);
        if (lane == 0) {
            const float count = static_cast<float>(arg6 * 4);
            const float mean = sum / count;
            const float variance = fmaxf(square_sum / count - mean * mean, 0.0f);
            warp_sums[0] = mean;
            warp_squares[0] = rsqrtf(variance + 1.0e-5f);
        }
    }
    __syncthreads();
    const float mean = warp_sums[0];
    const float inverse_stddev = warp_squares[0];

    for (int vector_index = threadIdx.x; vector_index < arg6;
         vector_index += threads) {
        const int scalar_index = vector_index * 4;
        const int local_row = scalar_index / (arg2 * arg4);
        const int remainder = scalar_index - local_row * arg2 * arg4;
        const int channel = remainder / arg4;
        const int column = remainder - channel * arg4;
        const float4 values = *reinterpret_cast<const float4*>(
            arg0 + batch * input_batch_stride + channel * arg3 * arg4
            + (group_begin + local_row) * arg4 + column);
        float4 result;
        result.x = (ceilf(values.x) - mean) * inverse_stddev;
        result.y = (ceilf(values.y) - mean) * inverse_stddev;
        result.z = (ceilf(values.z) - mean) * inverse_stddev;
        result.w = (ceilf(values.w) - mean) * inverse_stddev;
        *reinterpret_cast<float4*>(
            arg1 + batch * output_batch_stride
            + (group_begin + local_row) * arg2 * arg4
            + channel * arg4 + column) = result;
    }
}

extern "C" void launch_ceil_transpose_group_norm(
    const float* arg0, float* arg1, int arg2, int arg3, int arg4, int arg5,
    int arg6, cudaStream_t stream) {
    constexpr int threads = 256;
    const int group_height = arg4 / arg6;
    const int vectors_per_group = arg3 * group_height * arg5 / 4;

    ceil_transpose_group_norm_kernel<threads><<<
        dim3(static_cast<unsigned int>(arg6),
             static_cast<unsigned int>(arg2), 1),
        threads, 0, stream>>>(
            arg0, arg1, arg3, arg4, arg5,
            group_height, vectors_per_group);
}