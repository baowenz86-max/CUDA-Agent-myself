#include <cuda_runtime.h>

// Implement LayerNorm and Addcmul CUDA kernels and launchers
// These functions are declared in custom_ops_binding.cpp

__global__ void layer_norm_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int channels,
    int length,
    float eps
) {
    // Each block handles one normalization group: a single (batch, channel) pair
    // The layer norm normalizes over the entire (channel, length) dimension,
    // but here we interpret normalized_shape as (channels, length).
    // Wait: The contract says LayerNorm over the last dimension with normalized_shape equal to [out_channels, normalized_length].
    // That means it normalizes over both channels and length? Let's check reference:
    // In model.py, self.layer_norm = nn.LayerNorm(layer_norm_shape) where layer_norm_shape = (out_channels, out_length).
    // So it normalizes over the last two dimensions: channels and length.
    // Actually input shape is (batch, channels, length), and nn.LayerNorm((channels, length)) normalizes over the last two dims (channels and length) for each batch.
    // So each batch element is normalized independently over all channels and all length elements.
    // Thus we need one thread block per batch element, computing mean and variance over channels*length elements,
    // then applying weight and bias (which are per-element in the normalized shape).
    // However, the launcher signature has batch_size, channels, length. Let's implement accordingly.
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    int group_size = channels * length;
    const float* input_batch = input + batch * group_size;
    float* output_batch = output + batch * group_size;

    // Compute mean and variance using block reduction
    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_var = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Grid-stride loop over group_size with blockDim threads
    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = input_batch[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Reduce within block
    s_mean[tid] = local_sum;
    s_var[tid] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / group_size;
    float var = s_var[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Apply normalization and affine transformation
    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = input_batch[i];
        float normalized = (val - mean) * inv_std;
        output_batch[i] = normalized * weight[i] + bias[i];
    }
}

__global__ void addcmul_kernel(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] + tensor1[idx] * tensor2[idx];
    }
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
    int group_size = channels * length;
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem_size = 2 * threads * sizeof(float); // s_mean and s_var
    layer_norm_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        output, input, weight, bias, batch_size, channels, length, eps
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
    addcmul_kernel<<<blocks, threads, 0, stream>>>(
        output, input, tensor1, tensor2, numel
    );
}
