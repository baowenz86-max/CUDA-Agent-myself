#include <cuda_runtime.h>
#include <float.h>

// Elementwise absolute value: output[i] = fabsf(input[i])
__global__ void abs_kernel(const float* __restrict__ input, float* __restrict__ output, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        output[idx] = fabsf(input[idx]);
    }
}

// Softmax along dimension 1 (height) for a 3D tensor [batch, height, width].
// Input is contiguous: stride_batch = height * width, stride_height = width, stride_width = 1.
// Each thread handles one (batch, width) pair, iterating over all height values.
__global__ void softmax_dim1_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    int batch, int height, int width) {
    int total_threads = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch * width; idx += total_threads) {
        int b = idx / width;
        int w = idx % width;
        
        const float* in_base = input + (long long)b * height * width + w;
        float* out_base = output + (long long)b * height * width + w;
        
        // 1. Compute max for numerical stability
        float max_val = -FLT_MAX;
        for (int h = 0; h < height; ++h) {
            float val = in_base[(long long)h * width];
            if (val > max_val) max_val = val;
        }
        
        // 2. Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            float val = __expf(in_base[(long long)h * width] - max_val);
            sum += val;
            // Temporarily store exp(x-max) in output for step 3
            out_base[(long long)h * width] = val;
        }
        
        // 3. Normalize: output = exp(x-max) / sum
        float inv_sum = 1.0f / sum;
        for (int h = 0; h < height; ++h) {
            out_base[(long long)h * width] *= inv_sum;
        }
    }
}

// Launcher for elementwise abs
extern "C" void custom_abs_launcher(const float* input, float* output, long long total, cudaStream_t stream) {
    if (total <= 0) return;
    const int block_size = 256;
    // grid-stride loop -> fixed number of blocks based on SM count, sufficient to saturate
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, abs_kernel, block_size, 0);
    int grid_size = sm_count * max_blocks_per_sm;
    if (grid_size <= 0) grid_size = 256;
    // Clamp to avoid excessive blocks, but ensure enough parallelism
    if (grid_size > (total + block_size - 1) / block_size) grid_size = (int)((total + block_size - 1) / block_size);
    abs_kernel<<<grid_size, block_size, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, we would throw an exception; here we just note it.
        // The binding will check error and throw.
    }
}

// Launcher for softmax along dim=1
extern "C" void custom_softmax_dim1_launcher(const float* input, float* output,
                                             int batch, int height, int width, cudaStream_t stream) {
    if (batch <= 0 || height <= 0 || width <= 0) return;
    long long total_pairs = (long long)batch * width;
    const int block_size = 256;
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, softmax_dim1_kernel, block_size, 0);
    int grid_size = sm_count * max_blocks_per_sm;
    if (grid_size <= 0) grid_size = 256;
    // grid-stride loop, cap at necessary blocks
    if (grid_size > (total_pairs + block_size - 1) / block_size) grid_size = (int)((total_pairs + block_size - 1) / block_size);
    softmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(input, output, batch, height, width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked in binding.
    }
}
