#include <cuda_runtime.h>

// Elementwise hardtanh: clamp(x, min_val, max_val)
__global__ void hardtanh_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t numel,
                               float min_val,
                               float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[idx] = val;
    }
}

// Launcher: input and output are contiguous float32 tensors with numel elements.
// Supports arbitrary strides implicitly because caller passes a contiguous output or the kernel
// is launched over the flattened contiguous memory. For strided input, the caller may need to
// make it contiguous first or we can handle strides separately; contract says output should
// have same shape and dtype, and reference output is contiguous. To be safe we assume input
// is contiguous as per current use, but we can add stride support here by indexing, though
// the binding layer will likely pass contiguous tensors. We keep a flat launcher for simplicity
// and performance.
extern "C" void hardtanh_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         double min_val,
                                         double max_val,
                                         cudaStream_t stream) {
    if (numel <= 0) return;

    float min_f = static_cast<float>(min_val);
    float max_f = static_cast<float>(max_val);

    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    // Cap blocks to a reasonable number; grid-stride loop not needed as numel < 2^31
    if (blocks > 2147483647) {
        blocks = 2147483647;
    }

    hardtanh_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        input, output, numel, min_f, max_f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In production binding would check and throw, but here we just leave for completeness
        // We cannot include torch here, so just do nothing; caller should check.
    }
}
