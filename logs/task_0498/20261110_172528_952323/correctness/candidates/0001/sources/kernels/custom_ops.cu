#include <cuda_runtime.h>

// Custom CUDA kernels for model_new.py
// gelu_forward: element-wise GELU activation (approximate='none')
// prod_dim1_keepdim_forward: product along dimension 1 with keepdim=True

// GELU kernel: y = x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_forward_kernel(float* __restrict__ output,
                                    const float* __restrict__ input,
                                    long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        float x = input[i];
        float y = x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f)); // 1/sqrt(2)
        output[i] = y;
    }
}

// Product along dimension 1 with keepdim=True
// Input shape: [B, C, H, W] = [16, 128, 512, 256]
// Output shape: [B, 1, H, W]
// Each output location is product over C for a fixed (b, h, w).
// We use a two-phase approach:
// Phase 1: compute per-(b,h,w) product into a temporary contiguous buffer of size B*H*W.
// Phase 2: copy product into output tensor with shape [B,1,H,W] (which is same underlying size).
// This avoids atomics and preserves order (though product order is mathematically commutative).

// Phase 1 kernel: one thread per output location (or grid-stride over B*H*W)
__global__ void prod_dim1_partial_kernel(float* __restrict__ temp_product,
                                         const float* __restrict__ input,
                                         int B, int C, int H, int W) {
    long long total = (long long)B * H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total; i += stride) {
        // i corresponds to linear index over B,H,W
        int b = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;

        // Base index in input for dimension (b, 0, h, w)
        long long base = ((long long)b * C + 0) * H * W + (long long)h * W + w;
        float prod_val = input[base];
        long long channel_stride = (long long)H * W;
        for (int c = 1; c < C; ++c) {
            prod_val *= input[base + (long long)c * channel_stride];
        }
        temp_product[i] = prod_val;
    }
}

// Phase 2 kernel: copy temp_product to output with keepdim shape [B,1,H,W]
// Output tensor is also contiguous with B*1*H*W elements,
// so we can copy element-wise.
__global__ void copy_to_keepdim_kernel(float* __restrict__ output,
                                       const float* __restrict__ temp_product,
                                       long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        output[i] = temp_product[i];
    }
}

// Launcher functions (extern "C" so binding can call them)

extern "C" void gelu_forward_launcher(float* output,
                                      const float* input,
                                      long long total_elements,
                                      cudaStream_t stream) {
    if (total_elements == 0 || output == nullptr || input == nullptr) {
        return;
    }
    int threads = 256;
    // Cap grid to some reasonable max (e.g., 65535*64) and use grid-stride loop.
    // For total_elements up to 2^31, we can use 32-bit block count, but use long long for safety.
    long long blocks_ll = (total_elements + threads - 1) / threads;
    // Limit grid dimension to maximum allowed (2^31-1) but we can't launch that many blocks due to resource limits.
    // Use grid-stride with a max grid of 65535 * 64 = ~4.2M blocks, enough for parallelism.
    long long max_blocks = 65535LL * 64;
    if (blocks_ll > max_blocks) {
        blocks_ll = max_blocks;
    }
    int blocks = (int)blocks_ll;
    gelu_forward_kernel<<<blocks, threads, 0, stream>>>(output, input, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In binding layer, we could check and throw; here we just ignore but ensure no silent failure?
        // Ideally we would not have error handling in .cu, but we can leave this for correctness.
    }
}

extern "C" void prod_dim1_keepdim_launcher(float* output,
                                            const float* input,
                                            float* temp_product,
                                            int B, int C, int H, int W,
                                            cudaStream_t stream) {
    if (output == nullptr || input == nullptr || temp_product == nullptr || B == 0 || C == 0 || H == 0 || W == 0) {
        return;
    }
    long long total_locations = (long long)B * H * W;
    int threads = 256;
    long long blocks_ll = (total_locations + threads - 1) / threads;
    long long max_blocks = 65535LL * 64;
    if (blocks_ll > max_blocks) {
        blocks_ll = max_blocks;
    }
    int blocks = (int)blocks_ll;
    prod_dim1_partial_kernel<<<blocks, threads, 0, stream>>>(temp_product, input, B, C, H, W);

    // Copy kernel
    long long total_elements_out = (long long)B * 1 * H * W;
    blocks_ll = (total_elements_out + threads - 1) / threads;
    if (blocks_ll > max_blocks) {
        blocks_ll = max_blocks;
    }
    blocks = (int)blocks_ll;
    copy_to_keepdim_kernel<<<blocks, threads, 0, stream>>>(output, temp_product, total_elements_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore in .cu, binding can check after call
    }
}
