#include <cuda_runtime.h>

// LayerNorm forward kernel: normalizes over the last two dimensions
// Input layout: [N, C, L] where C*L is the normalization size
// Each block handles one sample (one N index)
// Grid dimension: N (number of batch elements)
// Block dimension: blockDim.x threads, shared memory for reduction
__global__ void layer_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int C,
    int L,
    float eps
) {
    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var = shared + blockDim.x;

    int n = blockIdx.x;
    int tid = threadIdx.x;
    int total = C * L;

    // Each thread processes multiple elements with grid-stride within the sample
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        // input is [N, C, L] potentially non-contiguous, need strides passed by launcher
        // We'll assume launcher makes input contiguous or passes base pointer with appropriate offsets
        // For safety, launcher should ensure contiguous or handle strides
        float val = input[n * total + i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store per-warp partials to shared memory
    int warpId = tid / 32;
    int laneId = tid % 32;
    int numWarps = blockDim.x / 32;
    if (laneId == 0) {
        s_mean[warpId] = sum;
        s_var[warpId] = sum_sq;
    }
    __syncthreads();

    // First warp reduces partials
    if (tid < 32) {
        sum = (tid < numWarps) ? s_mean[tid] : 0.0f;
        sum_sq = (tid < numWarps) ? s_var[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (tid == 0) {
            s_mean[0] = sum;
            s_var[0] = sum_sq;
        }
    }
    __syncthreads();

    float mean = s_mean[0] / total;
    float variance = s_var[0] / total - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Apply normalization and affine transform
    for (int i = tid; i < total; i += blockDim.x) {
        int idx = n * total + i;
        float val = (input[idx] - mean) * inv_std;
        int c = i / L;
        int l = i % L;
        output[idx] = val * weight[c * L + l] + bias[c * L + l];
    }
}

// Launcher for layer_norm_forward
// Assumes input is contiguous; if not, binding should make contiguous or provide strides.
// Since contract says supports strides, we need to handle non-contiguous by copying to contiguous
// or by passing strides. For simplicity, launcher will accept contiguous pointers only;
// binding layer will ensure contiguity before calling.
void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int L,
    float eps,
    cudaStream_t stream
) {
    int total = C * L;
    // Choose block size: 256 threads
    int threads = 256;
    // Shared memory size: 2 arrays of numWarps floats for reduction
    int numWarps = threads / 32;
    int shared_mem_size = 2 * numWarps * sizeof(float);

    // Grid: one block per sample
    dim3 grid(N);
    dim3 block(threads);

    layer_norm_kernel<<<grid, block, shared_mem_size, stream>>>(
        output, input, weight, bias, C, L, eps
    );
}
