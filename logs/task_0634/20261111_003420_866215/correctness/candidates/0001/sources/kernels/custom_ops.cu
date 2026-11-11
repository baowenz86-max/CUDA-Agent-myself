#include <cuda_runtime.h>
#include <cstdint>

// Helper function to get CUDA stream is defined in binding.cpp, but we only need cudaStream_t parameter in launcher

// =========================== to_cuda ===========================
// Copy tensor from CPU to GPU, preserving shape and stride.
// Launcher signature: void to_cuda_launcher(const float* cpu_input, float* gpu_output, int64_t numel, cudaStream_t stream)
extern "C" void to_cuda_launcher(const float* cpu_input, float* gpu_output, int64_t numel, cudaStream_t stream) {
    cudaMemcpyAsync(gpu_output, cpu_input, numel * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// =========================== mean_reduce_keepdim ===========================
// Compute mean over dims 1,2,3 for each batch element. Output shape [batch,1,1,1].
// Input: x (float*), batch_size (int), channels (int), height (int), width (int)
// Output: mean (float*), length batch_size
// Launcher signature: void mean_reduce_keepdim_launcher(const float* x, float* mean, int batch_size, int channels, int height, int width, cudaStream_t stream)
extern "C" void mean_reduce_keepdim_launcher(const float* x, float* mean, int batch_size, int channels, int height, int width, cudaStream_t stream) {
    // Each block handles one batch element, reducing over channels*height*width elements
    // Block size: 256 threads
    int block_size = 256;
    int grid_size = batch_size;
    
    // Kernel launch
    mean_reduce_kernel<<<grid_size, block_size, 0, stream>>>(x, mean, channels, height, width);
}

// Helper kernel for mean reduction: each block reduces all elements of one batch
__global__ void mean_reduce_kernel(const float* __restrict__ x, float* __restrict__ mean, int channels, int height, int width) {
    extern __shared__ float shared_sum[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    long long total_elems = (long long)channels * height * width;
    const float* base = x + batch_idx * total_elems;
    
    // Grid-stride loop within the block for reduction
    float local_sum = 0.0f;
    for (long long i = tid; i < total_elems; i += block_dim) {
        local_sum += base[i];
    }
    
    // Shared memory reduction
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction tree
    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean[batch_idx] = shared_sum[0] / (float)total_elems;
    }
}

// =========================== where_less_zero ===========================
// For each element, if x[i] < mean[batch_index], output 0.0, else output x[i].
// Input: x (float*), mean (float*), batch_size, channels, height, width
// Output: output (float*), same shape as x
// Launcher signature: void where_less_zero_launcher(const float* x, const float* mean, float* output, int batch_size, int channels, int height, int width, cudaStream_t stream)
extern "C" void where_less_zero_launcher(const float* x, const float* mean, float* output, int batch_size, int channels, int height, int width, cudaStream_t stream) {
    int total_elems = batch_size * channels * height * width;
    int block_size = 256;
    int grid_size = (total_elems + block_size - 1) / block_size;
    where_less_zero_kernel<<<grid_size, block_size, 0, stream>>>(x, mean, output, batch_size, channels, height, width);
}

__global__ void where_less_zero_kernel(const float* __restrict__ x, const float* __restrict__ mean, float* __restrict__ output, int batch_size, int channels, int height, int width) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elems = (long long)batch_size * channels * height * width;
    if (idx >= total_elems) return;
    
    long long batch_idx = idx / ((long long)channels * height * width);
    float threshold = mean[batch_idx];
    float val = x[idx];
    output[idx] = (val < threshold) ? 0.0f : val;
}
