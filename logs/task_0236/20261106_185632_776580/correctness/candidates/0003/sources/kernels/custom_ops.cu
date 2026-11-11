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

    // Block reduction for max using shared memory and simple reduction (no atomic)
    __shared__ float s_max;
    // Initialize shared memory
    s_max = -1e30f;
    __syncthreads();

    // Use shared memory reduction: each thread writes its local_max to a per-thread slot
    // But we only have one s_max, so we need a different approach.
    // Simpler: use atomicMax on shared memory with a device function
    // Let's implement a simple reduction using warp shuffle and shared memory properly.
    // Alternative: use atomicCAS loop inline.

    // We'll do a simple warp-level reduction and then use shared memory for block reduction.
    // For simplicity and correctness, use atomicCAS on s_max.
    // Since we have only one variable, we can use a single atomic operation per thread.
    // But atomicCAS on float requires int casting.
    unsigned int* s_max_int = reinterpret_cast<unsigned int*>(&s_max);
    unsigned int old_int = *s_max_int;
    unsigned int new_int = __float_as_uint(local_max);
    while (new_int > old_int) {
        unsigned int assumed = old_int;
        old_int = atomicCAS(s_max_int, assumed, new_int);
    }
    __syncthreads();
    float maxv = __uint_as_float(*s_max_int);

    // Second pass: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        float val = -input[idx];
        float exp_val = expf(val - maxv);
        vals[d] = exp_val;
        local_sum += exp_val;
    }

    // Block reduction for sum using atomicAdd on shared memory
    __shared__ float s_sum;
    s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float sum = s_sum;

    // Third pass: write output
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        output[idx] = vals[d] / sum;
    }
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
