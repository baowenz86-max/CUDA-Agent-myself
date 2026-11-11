#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Phase 1: Each block computes a partial sum over its assigned chunk of the plane using double precision.
// Input: float plane of size H*W, output: double partial sums per block (num_blocks_per_output values per output).
// grid = (num_outputs, num_blocks_per_output), block = 256 threads.
__global__ void reduce_plane_partial(const float* __restrict__ input,
                                     double* __restrict__ partial_sums,
                                     int H, int W, int plane_size,
                                     int num_blocks_per_output) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int out_idx = blockIdx.x;
    unsigned int block_in_output = blockIdx.y;

    const float* plane = input + (size_t)out_idx * plane_size;

    // Determine this block's contiguous chunk
    unsigned int chunk_size = (plane_size + num_blocks_per_output - 1) / num_blocks_per_output;
    unsigned int start = block_in_output * chunk_size;
    unsigned int end = min(start + chunk_size, (unsigned int)plane_size);

    double local_sum = 0.0;
    // Sequential strided loop within block; accumulation in double
    for (unsigned int idx = start + tid; idx < end; idx += blockDim.x) {
        local_sum += (double)plane[idx];
    }

    // Block reduction in double precision
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes partial sum to correct location
    if (tid == 0) {
        partial_sums[out_idx * num_blocks_per_output + block_in_output] = sdata[0];
    }
}

// Phase 2: Reduce partial sums for each output. One block per output.
// Each block has num_blocks_per_output partials, stored contiguously.
// We use a single warp (or all threads) to sum in order deterministically.
__global__ void reduce_plane_final(const double* __restrict__ partial_sums,
                                   float* __restrict__ output,
                                   int num_outputs,
                                   int num_blocks_per_output) {
    unsigned int out_idx = blockIdx.x;
    if (out_idx >= num_outputs) return;

    const double* partials = partial_sums + (size_t)out_idx * num_blocks_per_output;
    double sum = 0.0;
    for (int i = 0; i < num_blocks_per_output; ++i) {
        sum += partials[i];
    }
    output[out_idx] = (float)sum;
}

extern "C" void reduce_sum_launcher(const float* input, float* output,
                                   int N, int C, int H, int W,
                                   double* temp_partials,
                                   cudaStream_t stream) {
    int plane_size = H * W;
    int num_outputs = N * C;

    // Choose number of blocks per output; deterministic order requires fixed count.
    int num_blocks_per_output = 16;  // can tune later

    dim3 grid(num_outputs, num_blocks_per_output);
    int block_size = 256;
    size_t shared_mem_size = block_size * sizeof(double);

    // Phase 1
    reduce_plane_partial<<<grid, block_size, shared_mem_size, stream>>>(
        input, temp_partials, H, W, plane_size, num_blocks_per_output);

    // Phase 2: one block per output, say 128 threads (only one used for summing, but okay)
    int final_block_size = 128;
    reduce_plane_final<<<num_outputs, final_block_size, 0, stream>>>(
        temp_partials, output, num_outputs, num_blocks_per_output);
}

// Kernel for torch.round, round half to even
__global__ void round_kernel(const float* __restrict__ input, float* __restrict__ output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = rintf(input[idx]);
    }
}

extern "C" void round_launcher(const float* input, float* output, int numel, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    round_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
