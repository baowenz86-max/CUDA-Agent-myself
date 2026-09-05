#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// A block owns one (batch, group).  With width == 128, each warp owns a
// complete row, so both input loads and transposed-output stores are fully
// coalesced.  Advancing by eight rows per iteration avoids division and modulo
// in the bandwidth-critical loops.
__global__ __launch_bounds__(256, 2)
void ceil_transpose_group_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int height,
    int width,
    int group_height) {
    const int batch = blockIdx.y;
    const int h_begin = blockIdx.x * group_height;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rows = group_height * channels;
    const int input_batch_stride = channels * height * width;
    const int output_batch_stride = height * channels * width;
    const float* input_batch = input + batch * input_batch_stride;
    float* output_group = output + batch * output_batch_stride
                          + h_begin * channels * width;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int row = warp; row < rows; row += 8) {
        const int h_local = row >> 5;       // channels == 32
        const int channel = row & 31;
        const float4 value = *reinterpret_cast<const float4*>(
            input_batch + (channel * height + h_begin + h_local) * width
            + lane * 4);
        const float x0 = ceilf(value.x);
        const float x1 = ceilf(value.y);
        const float x2 = ceilf(value.z);
        const float x3 = ceilf(value.w);
        sum += (x0 + x1) + (x2 + x3);
        square_sum = fmaf(x0, x0, square_sum);
        square_sum = fmaf(x1, x1, square_sum);
        square_sum = fmaf(x2, x2, square_sum);
        square_sum = fmaf(x3, x3, square_sum);
    }

    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    __shared__ float partial_sum[8];
    __shared__ float partial_square_sum[8];
    if (lane == 0) {
        partial_sum[warp] = sum;
        partial_square_sum[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < 8 ? partial_sum[lane] : 0.0f;
        square_sum = lane < 8 ? partial_square_sum[lane] : 0.0f;
        sum = warp_sum(sum);
        square_sum = warp_sum(square_sum);
        if (lane == 0) {
            const float count = static_cast<float>(rows * width);
            const float mean = sum / count;
            const float variance = fmaxf(square_sum / count - mean * mean, 0.0f);
            partial_sum[0] = mean;
            partial_square_sum[0] = rsqrtf(variance + 1.0e-5f);
        }
    }
    __syncthreads();

    const float mean = partial_sum[0];
    const float inverse_std = partial_square_sum[0];
    for (int row = warp; row < rows; row += 8) {
        const int h_local = row >> 5;
        const int channel = row & 31;
        const float4 value = *reinterpret_cast<const float4*>(
            input_batch + (channel * height + h_begin + h_local) * width
            + lane * 4);
        float4 result;
        result.x = (ceilf(value.x) - mean) * inverse_std;
        result.y = (ceilf(value.y) - mean) * inverse_std;
        result.z = (ceilf(value.z) - mean) * inverse_std;
        result.w = (ceilf(value.w) - mean) * inverse_std;
        *reinterpret_cast<float4*>(output_group + row * width + lane * 4) = result;
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
    const dim3 grid(groups, batch);
    ceil_transpose_group_norm_kernel<<<grid, 256, 0, stream>>>(
        input, output, channels, height, width, height / groups);
}
