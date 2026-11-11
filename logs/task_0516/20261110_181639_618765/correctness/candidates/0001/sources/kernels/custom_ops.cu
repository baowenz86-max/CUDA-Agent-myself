#include <cuda_runtime.h>
#include <math.h>

// Group normalization kernel: normalizes input over (channels_per_group, height, width)
// Each block handles one (batch, group) pair, computing mean and variance then normalizing.
// Supports arbitrary strides via linear indexing using input strides provided by launcher.
// Since input may be non-contiguous, launcher passes strides; kernel uses offset calculation for each element.

__global__ void group_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_groups,
    int64_t channels_per_group,
    int64_t input_stride_batch,
    int64_t input_stride_channel,
    int64_t input_stride_height,
    int64_t output_stride_batch,
    int64_t output_stride_channel,
    int64_t output_stride_height,
    double eps
) {
    // blockIdx.x = batch index, blockIdx.y = group index
    int64_t batch = blockIdx.x;
    int64_t group = blockIdx.y;
    
    int64_t group_size = channels_per_group * height * width;
    int64_t tid = threadIdx.x;
    int64_t block_dim = blockDim.x;
    
    // First pass: compute sum and sum of squares using block reduction.
    // We'll use shared memory for reduction.
    extern __shared__ double sdata[];
    double* s_sum = sdata;
    double* s_sumsq = sdata + block_dim;
    
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (int64_t idx = tid; idx < group_size; idx += block_dim) {
        // Decompose idx into channel within group, h, w
        int64_t c_in_group = idx / (height * width);
        int64_t rem = idx % (height * width);
        int64_t h = rem / width;
        int64_t w = rem % width;
        int64_t channel = group * channels_per_group + c_in_group;
        int64_t input_offset = batch * input_stride_batch + channel * input_stride_channel + h * input_stride_height + w;
        float val = input[input_offset];
        local_sum += (double)val;
        local_sumsq += (double)val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();
    
    // Reduction (assuming block_dim is power of 2)
    for (int64_t stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }
    
    double mean = s_sum[0] / (double)group_size;
    double var = s_sumsq[0] / (double)group_size - mean * mean;
    double inv_std = rsqrt((double)var + eps);
    
    // Second pass: normalize each element
    for (int64_t idx = tid; idx < group_size; idx += block_dim) {
        int64_t c_in_group = idx / (height * width);
        int64_t rem = idx % (height * width);
        int64_t h = rem / width;
        int64_t w = rem % width;
        int64_t channel = group * channels_per_group + c_in_group;
        int64_t input_offset = batch * input_stride_batch + channel * input_stride_channel + h * input_stride_height + w;
        int64_t output_offset = batch * output_stride_batch + channel * output_stride_channel + h * output_stride_height + w;
        float norm_val = (float)(((double)input[input_offset] - mean) * inv_std);
        if (weight != nullptr) {
            norm_val *= weight[channel];
        }
        if (bias != nullptr) {
            norm_val += bias[channel];
        }
        output[output_offset] = norm_val;
    }
}

// Launcher function (C-linkage)
extern "C" void group_norm_forward_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_groups,
    const float* weight,
    const float* bias,
    double eps,
    int64_t input_stride_batch,
    int64_t input_stride_channel,
    int64_t input_stride_height,
    int64_t output_stride_batch,
    int64_t output_stride_channel,
    int64_t output_stride_height,
    cudaStream_t stream
) {
    int64_t channels_per_group = channels / num_groups;
    int64_t group_size = channels_per_group * height * width;
    
    // Choose block size: 256 threads, each block computes one (batch, group)
    int block_dim = 256;
    // Shared memory: two arrays of double size block_dim
    size_t shared_mem_size = 2 * block_dim * sizeof(double);
    
    dim3 grid((unsigned int)batch_size, (unsigned int)num_groups, 1);
    
    group_norm_kernel<<<grid, block_dim, shared_mem_size, stream>>>(
        output,
        input,
        weight,
        bias,
        batch_size,
        channels,
        height,
        width,
        num_groups,
        channels_per_group,
        input_stride_batch,
        input_stride_channel,
        input_stride_height,
        output_stride_batch,
        output_stride_channel,
        output_stride_height,
        eps
    );
}
