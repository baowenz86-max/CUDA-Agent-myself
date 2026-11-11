#include <cuda_runtime.h>

// Fused elementwise operation: y = sign(abs(x))
// For float32, sign returns -1.0 for negative, 0.0 for zero, 1.0 for positive.
// After abs, input is non-negative, so -1.0 is never produced for finite values.
// This kernel is equivalent to sign(x) for all finite x, and also maps -inf, NaN etc. as torch.sign would after abs.
__global__ void fused_abs_sign_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        float abs_val = fabsf(val);
        // sign(abs_val): 0.0 if abs_val == 0.0, 1.0 otherwise (since abs_val >= 0)
        // Use explicit comparison to handle -0.0 correctly (abs(-0.0)=0.0 -> sign 0.0)
        output[i] = (abs_val == 0.0f) ? 0.0f : 1.0f;
    }
}

// Launcher with C linkage, called from binding.cpp (not included here)
extern "C" void fused_abs_sign_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Grid-stride loop allows handling arbitrary numel; cap blocks to avoid excessive launch configuration if needed.
    // For large tensors, use a grid-stride loop with a reasonable number of blocks (e.g., 256 blocks or enough to cover).
    // We choose blocks = min(blocks, 65535 * 4) but actually grid stride handles all, we can use a reasonable max.
    // Simpler: use grid-stride but keep block count modest for occupancy.
    if (blocks > 1024) blocks = 1024; // limit for grid-stride, handles up to 1024*256 = 262144 elements per iteration, loop covers rest.
    fused_abs_sign_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}
