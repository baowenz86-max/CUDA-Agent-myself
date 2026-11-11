#include <cuda_runtime.h>
#include <cstdint>

// =========================== to_cuda ===========================
__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" void to_cuda_launcher(const float* cpu_input, float* gpu_output, int64_t numel, cudaStream_t stream) {
    int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    copy_kernel<<<grid_size, block_size, 0, stream>>>(cpu_input, gpu_output, (long long)numel);
}

// =========================== mean_reduce_keepdim ===========================
// Kernel definition before launcher
__global__ void mean_reduce_kernel(const float* __restrict__ x, float* __restrict__ mean, int channels, int height, int width) {
    extern __shared__ float shared_sum[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    long long total_elems = (long long)channels * height * width;
    const float* base = x + batch_idx * total_elems;
    
    float local_sum = 0.0f;
    for (long long i = tid; i < total_elems; i += block_dim) {
        local_sum += base[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
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

extern "C" void mean_reduce_keepdim_launcher(const float* x, float* mean, int batch_size, int channels, int height, int width, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = batch_size;
    int shared_mem_size = block_size * sizeof(float);
    mean_reduce_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x, mean, channels, height, width);
}

// =========================== where_less_zero ===========================
__global__ void where_less_zero_kernel(const float* __restrict__ x, const float* __restrict__ mean, float* __restrict__ output, int batch_size, int channels, int height, int width) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elems = (long long)batch_size * channels * height * width;
    if (idx >= total_elems) return;
    
    long long batch_idx = idx / ((long long)channels * height * width);
    float threshold = mean[batch_idx];
    float val = x[idx];
    output[idx] = (val < threshold) ? 0.0f : val;
}

extern "C" void where_less_zero_launcher(const float* x, const float* mean, float* output, int batch_size, int channels, int height, int width, cudaStream_t stream) {
    long long total_elems = (long long)batch_size * channels * height * width;
    int block_size = 256;
    long long grid_size = (total_elems + block_size - 1) / block_size;
    where_less_zero_kernel<<<grid_size, block_size, 0, stream>>>(x, mean, output, batch_size, channels, height, width);
}
