#include <cuda_runtime.h>

// LayerNorm kernel: input shape [batch, channels, length]
// weight and bias have shape [channels, length] (same as normalized_shape)
__global__ void layer_norm_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int channels,
    int length,
    float eps)
{
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int idx = threadIdx.x;

    // Each block handles one (batch, channel) pair.
    __shared__ float mean;
    __shared__ float var;
    __shared__ float inv_std;

    float thread_sum = 0.0f;
    for (int i = idx; i < length; i += blockDim.x) {
        float val = input[(batch * channels + channel) * length + i];
        thread_sum += val;
    }

    // Block reduction for sum
    __shared__ float sum_shared[256];
    sum_shared[idx] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            sum_shared[idx] += sum_shared[idx + stride];
        }
        __syncthreads();
    }
    if (idx == 0) {
        mean = sum_shared[0] / length;
    }
    __syncthreads();

    // Compute variance
    float thread_var_sum = 0.0f;
    for (int i = idx; i < length; i += blockDim.x) {
        float diff = input[(batch * channels + channel) * length + i] - mean;
        thread_var_sum += diff * diff;
    }
    __shared__ float var_shared[256];
    var_shared[idx] = thread_var_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            var_shared[idx] += var_shared[idx + stride];
        }
        __syncthreads();
    }
    if (idx == 0) {
        var = var_shared[0] / length;
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    // Normalize and apply affine
    for (int i = idx; i < length; i += blockDim.x) {
        float val = input[(batch * channels + channel) * length + i];
        float norm = (val - mean) * inv_std;
        // weight and bias are per (channel, i)
        float w = weight[channel * length + i];
        float b = bias[channel * length + i];
        output[(batch * channels + channel) * length + i] = norm * w + b;
    }
}

// Addcmul kernel: output = input + tensor1 * tensor2
template <typename scalar_t>
__global__ void addcmul_kernel(
    scalar_t* output,
    const scalar_t* input,
    const scalar_t* tensor1,
    const scalar_t* tensor2,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] + tensor1[idx] * tensor2[idx];
    }
}

// Launcher for layer_norm
extern "C" void layer_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int channels,
    int length,
    float eps,
    cudaStream_t stream)
{
    dim3 grid(batch_size, channels);
    dim3 block(256);
    layer_norm_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, batch_size, channels, length, eps);
}

// Launcher for addcmul
extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (numel + block_size - 1) / block_size;
    addcmul_kernel<float><<<grid_size, block_size, 0, stream>>>(
        output, input, tensor1, tensor2, numel);
}
