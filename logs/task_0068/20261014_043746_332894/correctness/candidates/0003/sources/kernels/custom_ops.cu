#include <cuda_runtime.h>
#include <float.h>

// ReLU forward kernel: elementwise max(x, 0)
__global__ void relu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// Softmax forward kernel: softmax over last dimension
// Each block handles one row (batch*channels*height rows)
// blockDim.x should be >= inner_size (256 threads covers 224)
__global__ void softmax_forward_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t outer_size,
                                       int64_t inner_size) {
    // Each block processes one row of length inner_size
    extern __shared__ float smem[];  // shared memory for the row
    
    int64_t row = blockIdx.x;
    if (row >= outer_size) return;
    
    const float* row_input = input + row * inner_size;
    float* row_output = output + row * inner_size;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Load row into shared memory and compute partial max
    float local_max = -FLT_MAX;
    for (int i = tid; i < inner_size; i += num_threads) {
        float val = row_input[i];
        smem[i] = val;
        local_max = fmaxf(local_max, val);
    }
    
    // Block reduction for max using shared memory
    __shared__ float max_shared[32]; // up to 32 warps (1024 threads)
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (num_threads + 31) / 32;
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) {
        max_shared[warp] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps (warp 0 only)
    if (warp == 0) {
        float block_max = (lane < num_warps) ? max_shared[lane] : -FLT_MAX;
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            max_shared[0] = block_max; // store final max
        }
    }
    __syncthreads();
    float max_val = max_shared[0]; // broadcast to all threads
    
    // Compute partial sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < inner_size; i += num_threads) {
        float val = __expf(smem[i] - max_val);
        local_sum += val;
    }
    
    // Block reduction for sum
    __shared__ float sum_shared[32];
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        sum_shared[warp] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (warp 0 only)
    if (warp == 0) {
        float block_sum = (lane < num_warps) ? sum_shared[lane] : 0.0f;
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            sum_shared[0] = block_sum; // store final sum
        }
    }
    __syncthreads();
    float sum = sum_shared[0];
    
    // Normalize and write output
    for (int i = tid; i < inner_size; i += num_threads) {
        row_output[i] = __expf(smem[i] - max_val) / sum;
    }
}

// Launcher for ReLU: output must be preallocated with same shape and stride, assumed contiguous
// total_elements = number of elements
// stream: CUDA stream
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total_elements,
                                      cudaStream_t stream) {
    if (total_elements <= 0) return;
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// Launcher for Softmax: input tensor shape [outer_size, inner_size] (contiguous)
// outer_size = product of all dims except last, inner_size = last dim
extern "C" void softmax_forward_launcher(const float* input,
                                         float* output,
                                         int64_t outer_size,
                                         int64_t inner_size,
                                         cudaStream_t stream) {
    if (outer_size <= 0 || inner_size <= 0) return;
    int threads = 256;
    // Ensure threads >= inner_size (usually 224, so 256 is fine)
    int shared_mem_size = inner_size * sizeof(float);
    // Limit grid to outer_size blocks
    softmax_forward_kernel<<<(int)outer_size, threads, shared_mem_size, stream>>>(
        input, output, outer_size, inner_size);
}
