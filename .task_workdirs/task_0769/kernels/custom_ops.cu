#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------------------
// Concatenation kernel: copy inputs into output along dim=1.
// inputs: array of pointers to each input tensor (each contiguous, same shape except dim=1)
// num_inputs: number of tensors to concatenate
// outer_size: number of elements before the concatenation dimension (batch in this task)
// inner_size: number of elements after the concatenation dimension (height*width in this task)
// concat_dim_size: size of each input along the concatenation dimension (channels per input)
// output: pointer to allocated contiguous output tensor
// Each thread copies one element from the appropriate input to output.
// We use a grid-stride loop over total output elements.
// -----------------------------------------------------------------------------
__global__ void concat_kernel(
    const float* const* __restrict__ inputs,
    float* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t concat_dim_size,
    int num_inputs,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // Compute indices: output has shape [outer_size, num_inputs*concat_dim_size, inner_size]
        int64_t tmp = linear / inner_size;
        int64_t inner_idx = linear % inner_size;
        int64_t out_channel = tmp % (num_inputs * concat_dim_size);
        int64_t outer = tmp / (num_inputs * concat_dim_size);
        // Determine which input and which channel within that input
        int input_idx = out_channel / concat_dim_size;
        int in_channel = out_channel % concat_dim_size;
        // Input index: outer * (concat_dim_size * inner_size) + in_channel * inner_size + inner_idx
        int64_t input_linear = (outer * concat_dim_size + in_channel) * inner_size + inner_idx;
        output[linear] = inputs[input_idx][input_linear];
    }
}

// -----------------------------------------------------------------------------
// Softmax kernel with grid-stride loop over positions.
// Input/output are contiguous tensors of shape [batch, channels, spatial].
// Each block processes multiple positions; blockDim.x covers channels.
// For each position, we compute max and sum reductions using shared memory.
// -----------------------------------------------------------------------------
__global__ void softmax_kernel_gridstride(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_positions,
    int64_t channels,
    int64_t spatial)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    for (int64_t pos = blockIdx.x; pos < total_positions; pos += gridDim.x) {
        int64_t b = pos / spatial;
        int64_t s = pos % spatial;
        int64_t base = (b * channels) * spatial + s;

        // Compute max over channels
        float max_val = -INFINITY;
        if (tid < channels) {
            max_val = input[base + tid * spatial];
        }
        smem[tid] = max_val;
        __syncthreads();
        for (int offset = block_size / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
            }
            __syncthreads();
        }
        max_val = smem[0];

        // Compute sum of exp(x - max)
        float sum_val = 0.0f;
        if (tid < channels) {
            float exp_val = expf(input[base + tid * spatial] - max_val);
            smem[tid] = exp_val;
        } else {
            smem[tid] = 0.0f;
        }
        __syncthreads();
        for (int offset = block_size / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                smem[tid] += smem[tid + offset];
            }
            __syncthreads();
        }
        sum_val = smem[0];

        // Write output
        if (tid < channels) {
            float exp_val = expf(input[base + tid * spatial] - max_val);
            output[base + tid * spatial] = exp_val / sum_val;
        }
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" so binding code can call them)
// -----------------------------------------------------------------------------

extern "C" void concat_forward_launcher(
    const float* const* inputs,
    int num_inputs,
    int64_t outer_size,
    int64_t inner_size,
    int64_t concat_dim_size,
    float* output,
    cudaStream_t stream)
{
    int64_t total_elements = outer_size * (num_inputs * concat_dim_size) * inner_size;
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    concat_kernel<<<blocks, threads, 0, stream>>>(
        inputs, output, outer_size, inner_size, concat_dim_size, num_inputs, total_elements);
}

extern "C" void softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t spatial,
    cudaStream_t stream)
{
    int64_t total_positions = batch * spatial;
    int block_size = 32;
    if (channels > block_size) block_size = 64; // fallback for larger channels
    int blocks = (int)std::min<int64_t>(total_positions, 65535);
    int shared_mem = block_size * sizeof(float);
    softmax_kernel_gridstride<<<blocks, block_size, shared_mem, stream>>>(
        input, output, total_positions, channels, spatial);
}
