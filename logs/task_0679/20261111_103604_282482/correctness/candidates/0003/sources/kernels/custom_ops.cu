#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// Kernel: sum over dimension 1 of input shape [B, C, H, W], with C=3.
// Output shape [B, H, W]. Input assumed contiguous.
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t batch,
                                int64_t height,
                                int64_t width) {
    int64_t total = batch * height * width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t b = i / (height * width);
        int64_t rem = i % (height * width);
        int64_t h = rem / width;
        int64_t w = rem % width;

        constexpr int C = 3;
        const float* base = input + b * C * height * width + h * width + w;
        float sum = base[0];
        sum += base[height * width];
        sum += base[2 * height * width];
        output[i] = sum;
    }
}

// Kernel: find maximum float value over all elements.
// Uses atomicMax on float bits to avoid NaN issues (we assume no NaN inputs).
__global__ void max_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ max_val,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = -FLT_MAX;
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        if (val > local_max) local_max = val;
    }

    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = __shfl_down_sync(0xffffffff, local_max, offset);
            if (other > local_max) local_max = other;
        }
    }

    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(local_max));
    }
}

// Kernel: find first index where input[i] == max_val.
__global__ void find_first_index_kernel(const float* __restrict__ input,
                                        const float* __restrict__ max_val,
                                        int64_t* __restrict__ output,
                                        int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    float target = *max_val;
    int64_t found_idx = -1;

    for (int64_t i = idx; i < numel; i += stride) {
        if (input[i] == target) {
            found_idx = i;
            break;
        }
    }

    // Only one thread should eventually write the result; use atomicMin to get the smallest index among all threads that found it.
    if (found_idx >= 0) {
        atomicMin(reinterpret_cast<unsigned long long*>(output), (unsigned long long)found_idx);
    }
}

// Launcher for sum reduction
extern "C" void torch_sum_dim1_launcher(const float* input,
                                         float* output,
                                         int64_t batch,
                                         int64_t height,
                                         int64_t width,
                                         cudaStream_t stream) {
    int64_t total = batch * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    blocks = (blocks < 1024) ? blocks : 1024;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, height, width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error will be checked by binding
    }
}

// Launcher for argmax over all elements
extern "C" void torch_argmax_all_launcher(const float* input,
                                           int64_t* output,
                                           int64_t numel,
                                           cudaStream_t stream) {
    // Allocate temporary device memory for max_val (float) and initialize.
    float* d_max_val;
    cudaMalloc(&d_max_val, sizeof(float));
    float init_max = -FLT_MAX;
    cudaMemcpyAsync(d_max_val, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    blocks = (blocks < 1024) ? blocks : 1024;

    // First pass: find max value
    max_reduce_kernel<<<blocks, threads, 0, stream>>>(input, d_max_val, numel);

    // Initialize output to a large value (e.g., numel) so atomicMin works.
    // Since valid indices are < numel, we set to numel (or LLONG_MAX).
    unsigned long long init_index = (unsigned long long)numel;
    cudaMemcpyAsync(output, &init_index, sizeof(unsigned long long), cudaMemcpyHostToDevice, stream);

    // Second pass: find first index
    find_first_index_kernel<<<blocks, threads, 0, stream>>>(input, d_max_val, output, numel);

    // Free temporary memory (asynchronously? use cudaFree after stream sync or using cudaFreeAsync? not available; we'll let binding sync).
    cudaFree(d_max_val);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // binding will check
    }
}
