// AdaptiveAvgPool1d CUDA kernel
// Supports arbitrary strides for non-contiguous input
#include <cuda_runtime.h>

// Kernel: each thread computes one output element
// Grid-stride loop over all output elements (N*C*output_size)
// We use int64 for large total elements, but here N*C*output_size <= 1M fits int32, but keep int64 for safety
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long N, long long C, long long L, long long output_size,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    long long total_outputs)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < total_outputs; linear += stride) {
        // decompose linear index into n, c, o
        long long o = linear % output_size;
        long long tmp = linear / output_size;
        long long c = tmp % C;
        long long n = tmp / C;
        
        // Adaptive average pooling window computation (PyTorch formula)
        long long start = (long long)((double)o * L / (double)output_size);
        long long end = (long long)ceil((double)(o + 1) * L / (double)output_size);
        if (end > L) end = L;
        long long window_size = end - start;
        if (window_size <= 0) {
            // Should not happen for valid output_size <= L
            output[n * out_stride_n + c * out_stride_c + o * out_stride_l] = 0.0f;
            continue;
        }
        
        const float* in_base = input + n * in_stride_n + c * in_stride_c;
        float sum = 0.0f;
        for (long long l = start; l < end; ++l) {
            sum += in_base[l * in_stride_l];
        }
        output[n * out_stride_n + c * out_stride_c + o * out_stride_l] = sum / (float)window_size;
    }
}

// Launcher: called from binding.cpp
// Input is assumed to be float32 CUDA tensor of shape [N,C,L]
// output_size is int64
// We require input pointer non-null, but binding layer checks
// Returns via preallocated output pointer
extern "C" void custom_adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t L, int64_t output_size,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_l,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_l,
    cudaStream_t stream)
{
    long long total_outputs = (long long)N * C * output_size;
    if (total_outputs <= 0) return;
    
    int threads = 256;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    // cap blocks to avoid grid dimension limit, grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    
    adaptive_avg_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        (long long)N, (long long)C, (long long)L, (long long)output_size,
        (long long)in_stride_n, (long long)in_stride_c, (long long)in_stride_l,
        (long long)out_stride_n, (long long)out_stride_c, (long long)out_stride_l,
        total_outputs);
    
    // Error check after launch (optional for performance, but good for debugging)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }  // binding can handle
}
