#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel to reduce a [H,W] plane of input to a scalar and add to output using atomicAdd.
// Each block processes a contiguous chunk of the plane.
// gridDim.x = number of output elements (batch*channels), blockIdx.x indexes output.
// gridDim.y = number of blocks per output element to achieve enough parallelism.
// blockDim.x = threads per block. Each thread strides over its assigned portion of the plane into shared memory, then block reduces.

__global__ void reduce_plane_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    int H, int W, int plane_size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int out_idx = blockIdx.x;
    unsigned int block_in_output = blockIdx.y;
    unsigned int num_blocks_per_output = gridDim.y;

    // Compute base pointer for this output's plane
    const float* plane = input + (size_t)out_idx * plane_size;

    float local_sum = 0.0f;

    // Each block processes a contiguous portion of the plane using grid-stride within the block
    // Determine the range [start, end) of elements this block is responsible for
    unsigned int chunk_size = (plane_size + num_blocks_per_output - 1) / num_blocks_per_output;
    unsigned int start = block_in_output * chunk_size;
    unsigned int end = min(start + chunk_size, (unsigned int)plane_size);

    // Each thread iterates over elements with stride blockDim.x within [start, end)
    for (unsigned int idx = start + tid; idx < end; idx += blockDim.x) {
        local_sum += plane[idx];
    }

    // Block reduction via shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds block result to output
    if (tid == 0) {
        atomicAdd(&output[out_idx], sdata[0]);
    }
}

// Launcher for reduce_sum over dims (2,3) of a 4D tensor [N, C, H, W]
// Input pointer is contiguous float32. Output preallocated zero of shape [N, C]
extern "C" void reduce_sum_launcher(const float* input, float* output,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream) {
    int plane_size = H * W;
    int num_outputs = N * C;

    // Choose number of blocks per output to balance occupancy and atomic contention.
    // For large H*W (589824), we want enough blocks to saturate GPU.
    // Each block can handle many elements; using 64 blocks per output gives 98304 threads total for 1536 outputs*64 blocks = 98304 blocks, maybe too many but OK.
    // We'll set num_blocks_per_output = 32, block size 256. That gives 1536*32 = 49152 blocks, 256 threads each => 12.5M threads, enough parallelism.
    // But too many blocks may cause atomic contention. We'll start with 16 blocks per output.
    int num_blocks_per_output = 16;
    dim3 grid(num_outputs, num_blocks_per_output);
    int block_size = 256;
    size_t shared_mem_size = block_size * sizeof(float);

    reduce_plane_kernel<<<grid, block_size, shared_mem_size, stream>>>(input, output, H, W, plane_size);
}

// Kernel for torch.round, round half to even
__global__ void round_kernel(const float* __restrict__ input, float* __restrict__ output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = rintf(input[idx]);
    }
}

// Launcher for round operation
extern "C" void round_launcher(const float* input, float* output, int numel, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    round_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
