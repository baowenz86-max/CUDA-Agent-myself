#include <cuda_runtime.h>

// Kernel for mean reduction over dims (1,2,3) with keepdim=True.
// Each block handles one batch element, iterating over the 12582912 float values.
// Input is float32 contiguous; output float32.
template <int BLOCK_SIZE>
__global__ void mean_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int batch_size,
                                  long long elements_per_batch) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* batch_ptr = input + (long long)batch_idx * elements_per_batch;
    float sum = 0.0f;

    // Grid-stride loop within this block for the current batch element
    for (long long i = threadIdx.x; i < elements_per_batch; i += BLOCK_SIZE) {
        sum += batch_ptr[i];
    }

    // Shared memory reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch_idx] = shared_sum[0] / (float)elements_per_batch;
    }
}

// Kernel for elementwise tangent. Each thread handles one element.
__global__ void tan_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = tanf(input[idx]);
    }
}

// Launcher for custom_mean_reduce.
// input: contiguous float32 CUDA tensor with shape [batch_size, 3, 2048, 2048]
// output: contiguous float32 CUDA tensor with shape [batch_size, 1, 1, 1]
// batch_size: 128
extern "C" void custom_mean_reduce_launcher(const float* input,
                                           float* output,
                                           int batch_size,
                                           long long elements_per_batch,
                                           cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = batch_size; // one block per batch
    mean_reduce_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        input, output, batch_size, elements_per_batch);
}

// Launcher for custom_tan.
// input/output: contiguous float32 CUDA tensor with numel elements
extern "C" void custom_tan_launcher(const float* input,
                                    float* output,
                                    int numel,
                                    cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    tan_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
