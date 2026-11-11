#include <cuda_runtime.h>

// Softmax along dimension 1 (channels) for 3D tensor [batch, channels, length]
// Input and output are contiguous float32 tensors.
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int length)
{
    // Each thread block handles one (batch, length) position, reducing over channels.
    // blockIdx.x = batch*length + length_idx, or use 2D grid for clarity.
    int total_positions = batch * length;
    int pos = blockIdx.x;
    if (pos >= total_positions) return;

    int b = pos / length;
    int l = pos % length;

    const float* in_ptr = input + b * channels * length + l;
    float* out_ptr = output + b * channels * length + l;

    // Compute max over channels using block reduction.
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + blockDim.x;

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    for (int c = tid; c < channels; c += blockDim.x) {
        float val = in_ptr[c * length];
        if (val > local_max) local_max = val;
    }

    // Block reduction for max
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }
    float block_max = s_max[0];

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int c = tid; c < channels; c += blockDim.x) {
        float val = in_ptr[c * length];
        float exp_val = expf(val - block_max);
        local_sum += exp_val;
        // Store exp values in shared memory for second pass (optional, we recompute)
        // To avoid recomputing exp, we can write to shared memory but channels may be large.
        // Recomputing exp is acceptable for now.
    }

    // Block reduction for sum
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float block_sum = s_sum[0];

    // Normalize and write output
    float inv_sum = 1.0f / block_sum;
    for (int c = tid; c < channels; c += blockDim.x) {
        float val = in_ptr[c * length];
        out_ptr[c * length] = expf(val - block_max) * inv_sum;
    }
}

// ConstantPad1d kernel: pads the last dimension with a constant value on both sides.
// padding is the number of elements to add on each side (same left and right).
__global__ void pad_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int length,
    int padding,
    float value)
{
    int output_length = length + 2 * padding;
    int total_elements = batch * channels * output_length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int l = idx % output_length;
    int rem = idx / output_length;
    int c = rem % channels;
    int b = rem / channels;

    if (l < padding || l >= padding + length) {
        output[idx] = value;
    } else {
        int input_l = l - padding;
        int input_idx = (b * channels + c) * length + input_l;
        output[idx] = input[input_idx];
    }
}

extern "C" void softmax_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    cudaStream_t stream)
{
    int total_positions = batch * length;
    int threads = 256;
    int blocks = total_positions;
    size_t shared_mem_size = 2 * threads * sizeof(float);

    softmax_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        input, output, batch, channels, length);
}

extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    int padding,
    double value,
    cudaStream_t stream)
{
    int output_length = length + 2 * padding;
    int total_elements = batch * channels * output_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    float value_float = (float)value;

    pad_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, length, padding, value_float);
}
