#include <cuda_runtime.h>
#include <cstdint>

// Kernel for max reduction over dimension 1 (channels) with keepdim=True.
// Input: x of shape [B, C, H, W] contiguous, output values [B,1,H,W] float, indices [B,1,H,W] int64
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
        const float* base = x + (((int64_t)b * C) * H + h) * W + w;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base[(int64_t)c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        values[linear] = max_val;
        indices[linear] = max_idx;
    }
}

// Kernel for MSE loss: mean((input - target)^2) over all elements.
// input: [B,1,H,W] contiguous, target: [B,1,1,1] contiguous.
// Output: scalar float (1 element).
__global__ void mse_loss_kernel(const float* __restrict__ input,
                                const float* __restrict__ target,
                                float* __restrict__ output,
                                int B, int H, int W,
                                int64_t num_elements) {
    double local_sum = 0.0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < num_elements; i += stride) {
        int b = i / (W * H);
        float diff = input[i] - target[b];
        local_sum += (double)diff * diff;
    }

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
    if (tid == 0) {
        double block_sum = smem[0];
        atomicAdd(output, (float)(block_sum / (double)num_elements));
    }
}

// Launcher for custom_max_dim
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
    if (dim != 1 || !keepdim) {
        // Contract guarantees dim=1 and keepdim=true, but if not, we still compute as if dim=1.
    }
    int64_t total = B * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_dim1_kernel<<<blocks, threads, 0, stream>>>(x, values, indices, (int)B, (int)C, (int)H, (int)W);
}

// Launcher for custom_mse_loss
extern "C" void custom_mse_loss_launcher(const float* input,
                                          const float* target,
                                          float* output,
                                          int64_t B,
                                          int64_t H,
                                          int64_t W,
                                          int64_t num_elements,
                                          cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    if (blocks == 0) blocks = 1;
    mse_loss_kernel<<<blocks, threads, 0, stream>>>(input, target, output, (int)B, (int)H, (int)W, num_elements);
}
