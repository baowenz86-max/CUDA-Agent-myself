#include <cuda_runtime.h>

// ============================================================================
// relu_transform: elementwise ReLU, output = max(input, 0)
// Supports shape [512, 512, 512] float32 contiguous tensor.
// ============================================================================
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

// ============================================================================
// smooth_l1_loss_mean_zero: computes mean over all elements of
//   if |z| < 1: 0.5 * z^2
//   else: |z| - 0.5
// where z is input element, target is zero, beta=1.0, reduction=mean.
// Output is scalar tensor (numel=1).
// ============================================================================
__global__ void smooth_l1_partial_sum_kernel(const float* __restrict__ input,
                                             float* __restrict__ partial_sums,
                                             long long numel) {
    extern __shared__ float sdata[];
    long long tid = threadIdx.x;
    long long idx = blockIdx.x * (long long)blockDim.x + tid;
    long long stride = (long long)gridDim.x * blockDim.x;
    float local_sum = 0.0f;
    for (long long i = idx; i < numel; i += stride) {
        float z = input[i];
        float abs_z = fabsf(z);
        float loss = (abs_z < 1.0f) ? 0.5f * z * z : abs_z - 0.5f;
        local_sum += loss;
    }

    // Block-level reduction using shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void final_reduce_mean_kernel(const float* __restrict__ partial_sums,
                                         float* __restrict__ output,
                                         int num_blocks,
                                         long long total_elements) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = sdata[0] / static_cast<float>(total_elements);
    }
}

// ============================================================================
// Launchers with C-linkage
// ============================================================================
extern "C" void relu_transform_launcher(const float* input,
                                         float* output,
                                         long long numel,
                                         cudaStream_t stream) {
    const int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    relu_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, numel);
}

extern "C" void smooth_l1_loss_mean_zero_launcher(const float* input,
                                                  float* output,
                                                  long long numel,
                                                  cudaStream_t stream) {
    // Configuration for partial sum kernel
    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    if (grid_size > 65535) {
        grid_size = 65535;  // cap grid size for manageable partial sum count
    }

    // Allocate partial sums on device
    float* d_partial_sums = nullptr;
    cudaMalloc(&d_partial_sums, grid_size * sizeof(float));

    // Launch partial sum kernel with shared memory for block reduction
    size_t shared_mem = block_size * sizeof(float);
    smooth_l1_partial_sum_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        input, d_partial_sums, numel);

    // Launch final reduction kernel to combine partial sums and compute mean
    int final_block_size = 256;
    size_t final_shared_mem = final_block_size * sizeof(float);
    final_reduce_mean_kernel<<<1, final_block_size, final_shared_mem, stream>>>(
        d_partial_sums, output, grid_size, numel);

    // Free temporary buffer after kernel completion (stream ordered)
    cudaFree(d_partial_sums);
}
