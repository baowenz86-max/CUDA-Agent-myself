#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel to reduce a [H,W] plane of input to a scalar and write to output.
// One block per output element, deterministically sums the plane in double precision.
// blockIdx.x = output index (0..N*C-1), blockDim.x = 256 threads.
// Each thread accumulates over indices start = tid, step = blockDim.x within the plane.
// Then block reduces via shared memory, thread 0 writes float output.
__global__ void reduce_plane_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    int H, int W, int plane_size) {
    extern __shared__ double sdata_double[];
    unsigned int tid = threadIdx.x;
    unsigned int out_idx = blockIdx.x;

    const float* plane = input + (size_t)out_idx * plane_size;

    double local_sum = 0.0;

    // Each thread loops over its assigned elements with stride blockDim.x
    for (unsigned int idx = tid; idx < plane_size; idx += blockDim.x) {
        local_sum += (double)plane[idx];
    }

    // Store local double partial in shared memory
    sdata_double[tid] = local_sum;
    __syncthreads();

    // Block reduction using double
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_double[tid] += sdata_double[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum as float
    if (tid == 0) {
        output[out_idx] = (float)sdata_double[0];
    }
}

// Launcher for reduce_sum over dims (2,3) of a 4D tensor [N, C, H, W]
// Input pointer is contiguous float32. Output preallocated zero of shape [N, C]
extern "C" void reduce_sum_launcher(const float* input, float* output,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream) {
    int plane_size = H * W;
    int num_outputs = N * C;

    // One block per output, 256 threads, shared memory for double reduction
    int block_size = 256;
    size_t shared_mem_size = block_size * sizeof(double);

    reduce_plane_kernel<<<num_outputs, block_size, shared_mem_size, stream>>>(input, output, H, W, plane_size);
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
