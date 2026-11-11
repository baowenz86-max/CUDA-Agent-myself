#include <cuda_runtime.h>
#include <cfloat>

// Softmin along dimension 1 for a 3D tensor with shape [B, D, H].
// Input is assumed contiguous float32 CUDA tensor.
// Output has same shape and dtype.
// Formula: softmin(x)[b, :, h] = exp(-x[b, :, h]) / sum_{d} exp(-x[b, d, h])
// Implemented as softmax of negative input with numerical stability.

__global__ void softmin_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int B, int D, int H) {
    int b = blockIdx.y;
    int h = blockIdx.x;
    if (b >= B || h >= H) return;

    extern __shared__ float smem[];
    float* vals = smem;        // D floats for exp values
    float* max_shared = &smem[D]; // block-level max
    float* sum_shared = &smem[D + 1]; // block-level sum (we can reuse max_shared after? but we need both, so we keep separate)
    // Actually we need only one scalar for max and one for sum; we can store them in shared.

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // First pass: find max over D for stability.
    // Use a proper reduction: each thread computes local max over its elements,
    // then we do block reduction using shared memory and warp shuffles.
    float local_max = -FLT_MAX;
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        float val = -input[idx];
        if (val > local_max) local_max = val;
    }

    // Block reduction for max using warp shuffle and shared memory
    // Standard reduction pattern.
    // Store local_max in shared memory
    if (tid < blockDim.x) {
        smem[tid] = local_max;
    }
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }

    float maxv = smem[0];

    // Second pass: compute exp(val - max) and sum
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        float val = -input[idx];
        float exp_val = expf(val - maxv);
        vals[d] = exp_val;
        local_sum += exp_val;
    }

    // Block reduction for sum using shared memory
    // Store local_sum
    if (tid < blockDim.x) {
        smem[tid] = local_sum;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    float sum = smem[0];

    // Third pass: write output
    for (int d = tid; d < D; d += stride) {
        int idx = (b * D + d) * H + h;
        output[idx] = vals[d] / sum;
    }
}

// Launcher function: out is preallocated by binding layer, same shape as input.
extern "C" void softmin_dim1_launcher(const float* input, float* output,
                                      int B, int D, int H,
                                      cudaStream_t stream) {
    if (input == nullptr || output == nullptr) return;

    dim3 grid(H, B);
    int block_size = 256;
    // Dynamic shared memory: D floats for vals + blockDim.x floats for reduction scratch + 2 floats for max/sum,
    // but we reuse smem for both vals and scratch. Need D + blockDim.x floats.
    // With block_size=256, D=512 -> 768 floats, 3KB.
    size_t shared_mem_size = (D + block_size + 2) * sizeof(float);

    softmin_dim1_kernel<<<grid, block_size, shared_mem_size, stream>>>(
        input, output, B, D, H);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // No print, but can handle if needed.
    }
}
