#include <cuda_runtime.h>

// Softmin along dimension 1 for a 3D tensor with shape [B, D, H].
// Input is assumed contiguous float32 CUDA tensor.
// Output has same shape and dtype.
// Formula: softmin(x)[b, :, h] = exp(-x[b, :, h]) / sum_{d} exp(-x[b, d, h])
// Implemented as softmax of negative input with numerical stability.

__global__ void softmin_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int B, int D, int H) {
    // Each block handles one (b, h) pair.
    int b = blockIdx.y;
    int h = blockIdx.x;
    if (b >= B || h >= H) return;

    // Shared memory for storing values and sum, size = 2*D floats
    extern __shared__ float smem[];
    float* vals = smem;
    float* max_val = &smem[D];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // First pass: find max over D for stability
    float local_max = -1e30f;
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        float val = -input[idx];
        if (val > local_max) local_max = val;
    }

    // Block reduction for max
    __shared__ float s_max;
    if (tid == 0) s_max = -1e30f;
    __syncthreads();
    atomicMax_block(&s_max, local_max);
    __syncthreads();
    float maxv = s_max;

    // Second pass: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        float val = -input[idx];
        float exp_val = expf(val - maxv);
        vals[d] = exp_val;
        local_sum += exp_val;
    }

    // Block reduction for sum
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd_block(&s_sum, local_sum);
    __syncthreads();
    float sum = s_sum;

    // Third pass: write output
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        output[idx] = vals[d] / sum;
    }
}

// Helper atomic reductions (float using atomicCAS)
__device__ void atomicMax_block(float* addr, float value) {
    float old = *addr;
    while (value > old) {
        float assumed = old;
        old = atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(value));
        old = __int_as_float(old);
    }
}

__device__ void atomicAdd_block(float* addr, float value) {
    atomicAdd(addr, value);
}

// Launcher function: out is preallocated by binding layer, same shape as input.
// input and output are raw pointers to contiguous float32 CUDA tensors.
extern "C" void softmin_dim1_launcher(const float* input, float* output,
                                      int B, int D, int H,
                                      cudaStream_t stream) {
    if (input == nullptr || output == nullptr) return;

    // Grid dimensions: blocks in x for H, in y for B
    dim3 grid(H, B);
    int block_size = 256;
    // Dynamic shared memory: D floats for vals, plus some for reduction (we use separate
    // variables but they occupy shared memory so need 2*D floats)
    size_t shared_mem_size = 2 * D * sizeof(float);

    softmin_dim1_kernel<<<grid, block_size, shared_mem_size, stream>>>(
        input, output, B, D, H);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Binding layer can handle error reporting, but we might want to print
        // or just let it fail. We'll not include any prints to keep minimal.
    }
}
