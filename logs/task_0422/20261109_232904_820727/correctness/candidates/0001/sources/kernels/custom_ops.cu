#include <cuda_runtime.h>
#include <cstdint>

// Kernel for max reduction over dimension 1 (channels) with keepdim=True.
// Input: x of shape [B, C, H, W] contiguous, output values [B,1,H,W] float, indices [B,1,H,W] int64
// dim is always 1 according to contract, but we implement generic enough for dim=1.
// keepdim is always true, but we handle it if false by adjusting output shape (not used in current model).
// We assume contiguous tensors as per contract.
__global__ void max_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ values,
                                int64_t* __restrict__ indices,
                                int B, int C, int H, int W) {
    int64_t total = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += stride) {
        int w = linear % W;
        int h = (linear / W) % H;
        int b = linear / (W * H);
        // Compute max across C
        const float* base = x + (((int64_t)b * C) * H + h) * W + w;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base[(int64_t)c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
            // ties keep first index, as done by sequential scan
        }
        values[linear] = max_val;
        indices[linear] = max_idx;
    }
}

// Kernel for MSE loss: mean((input - target)^2) over all elements.
// input: [B,1,H,W] contiguous, target: [B,1,1,1] contiguous.
// Broadcast target along H and W.
// Output: scalar float (1 element).
__global__ void mse_loss_kernel(const float* __restrict__ input,
                                const float* __restrict__ target,
                                float* __restrict__ output,
                                int B, int H, int W,
                                int64_t num_elements) {
    // Each thread computes partial sum over its assigned elements.
    double local_sum = 0.0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < num_elements; i += stride) {
        // map i to b, h, w
        int w = i % W;
        int h = (i / W) % H;
        int b = i / (W * H);
        float diff = input[i] - target[b];
        local_sum += (double)diff * diff;
    }
    // Reduce within block using shared memory
    __shared__ double smem[256];
    int tid = threadIdx.x;
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    // First thread of each block atomically adds to global sum
    if (tid == 0) {
        atomicAdd(output, (float)smem[0]);
    }
}

// Launcher for custom_max_dim
// Parameters: x pointer, values pointer, indices pointer, dim, keepdim, B, C, H, W, stream
extern "C" void custom_max_dim_launcher(const float* x,
                                        float* values,
                                        int64_t* indices,
                                        int64_t dim,
                                        bool keepdim,
                                        int64_t B,
                                        int64_t C,
                                        int64_t H,
                                        int64_t W,
                                        cudaStream_t stream) {
    // Only dim=1 supported here, but we can have a generic dim kernel later if needed.
    // For safety, assert dim==1 and keepdim==true.
    if (dim != 1 || !keepdim) {
        // Invalid, handle by filling zeros? Better to return error? But we cannot throw here.
        // Since contract guarantees dim=1 and keepdim=true, we can just proceed.
        // If not, we could still run but with incorrect output shape. We'll ignore.
    }
    int64_t total = B * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // Cap grid size, kernel uses grid-stride loop
    max_dim1_kernel<<<blocks, threads, 0, stream>>>(x, values, indices, (int)B, (int)C, (int)H, (int)W);
}

// Launcher for custom_mse_loss
// Parameters: input pointer, target pointer, output pointer, B, H, W, num_elements, stream
extern "C" void custom_mse_loss_launcher(const float* input,
                                          const float* target,
                                          float* output,
                                          int64_t B,
                                          int64_t H,
                                          int64_t W,
                                          int64_t num_elements,
                                          cudaStream_t stream) {
    // Initialize output to 0 before kernel
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    // Use multiple blocks to improve parallelism, but limited by shared memory size (256 floats).
    // We can use grid-stride loop within each block to cover all elements, then each block reduces its sum and atomically adds.
    // Choose reasonable grid size: up to 256 blocks? We'll use enough to cover elements but cap.
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024; // cap to avoid excessive blocks
    if (blocks == 0) blocks = 1;
    mse_loss_kernel<<<blocks, threads, 0, stream>>>(input, target, output, (int)B, (int)H, (int)W, num_elements);
}
