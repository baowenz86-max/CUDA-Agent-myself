#include <cuda_runtime.h>

// ConvTranspose1d kernel (forward)
// Each thread computes one output element: output[b, oc, ol] = bias[oc] + sum_{ic, k} input[b, ic, il] * weight[ic, oc, k]
// where il = (ol + padding - k) / stride must be integer and 0 <= il < in_length
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_length;
    if (idx >= total) return;

    int ol = idx % out_length;
    int oc = (idx / out_length) % out_channels;
    int b = idx / (out_length * out_channels);

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Iterate over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int il = ol + padding - k;
            if (il < 0) continue;
            if (il % stride != 0) continue;
            il /= stride;
            if (il >= 0 && il < in_length) {
                sum += input[((b * in_channels + ic) * in_length) + il] *
                       weight[((ic * out_channels + oc) * kernel_size) + k];
            }
        }
    }

    output[idx] = sum;
}

// LayerNorm kernel: normalize over last dimension (length) for each batch and channel,
// then apply elementwise affine weight[channel, :] and bias[channel, :].
// For simplicity, each block handles one batch*channel row; threadIdx.x maps to length.
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int length,
    float eps
) {
    extern __shared__ float smem[]; // [mean, variance]
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_size = length;
    int total_rows = batch_size * channels;
    if (row >= total_rows) return;

    const float* in_row = input + row * row_size;
    float* out_row = output + row * row_size;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        sum += in_row[i];
    }
    // Block reduction (simple, using shared memory)
    __shared__ float warp_sums[32];
    // Initialize shared reduction with warp sums
    sum = __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    // For simplicity, we compute mean using a single warp reduction per block
    // This assumes blockDim.x <= 32; we'll ensure blockDim.x = 256 in launcher but reduce appropriately.
    // Proper reduction: store partial sums in shared and reduce
    __shared__ float partial[256];
    if (tid < 256) partial[tid] = sum;
    __syncthreads();
    // Reduce partial sums (assuming blockDim.x <= 256)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    float mean = partial[0] / row_size;

    // Compute variance (sum of squared differences)
    float var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var_sum += diff * diff;
    }
    // Reduce var_sum
    if (tid < 256) partial[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    float variance = partial[0] / row_size;
    float inv_std = rsqrtf(variance + eps);

    // Apply normalization and affine
    for (int i = tid; i < row_size; i += blockDim.x) {
        float normalized = (in_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
    }
}

// Addcmul kernel: output = input + value * t1 * t2, with value = 1.0
__global__ void addcmul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ t1,
    const float* __restrict__ t2,
    float* __restrict__ output,
    int numel,
    float value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] + value * t1[idx] * t2[idx];
    }
}

// Launchers

extern "C" void conv_transpose_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
) {
    int total = batch_size * out_channels * out_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, padding
    );
}

extern "C" void layer_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int channels,
    int length,
    float eps,
    cudaStream_t stream
) {
    int rows = batch_size * channels;
    int threads = 256;
    int blocks = rows;
    size_t shared_mem = 2 * sizeof(float); // for mean, var (will store in partial but dynamic shared not needed for that)
    layer_norm_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, weight, bias, output,
        batch_size, channels, length, eps
    );
}

extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int numel,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    float value = 1.0f;
    addcmul_kernel<<<blocks, threads, 0, stream>>>(
        input, tensor1, tensor2, output, numel, value
    );
}
