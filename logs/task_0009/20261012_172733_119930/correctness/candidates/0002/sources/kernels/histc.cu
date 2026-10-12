#include <cuda_runtime.h>

// Kernel to compute histogram via atomic additions to integer bins (scaled for float output).
__global__ void histc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int bins,
    float min,
    float max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = input[idx];
    if (val < min || val > max) return;
    // NaN check: NaN is unordered, so val >= min && val <= max will be false if NaN? Actually comparisons with NaN return false, so NaN will be ignored automatically.
    // But to be explicit, we can skip if isnan(val)
    if (isnan(val)) return;

    float range = max - min;
    if (range <= 0.0f) return; // degenerate bins; reference may have behavior but treat as empty

    float bin_f = (val - min) / range * bins;
    int bin = (int)floorf(bin_f);
    if (bin >= bins) bin = bins - 1; // handle max exactly
    if (bin < 0) bin = 0;

    atomicAdd(&output[bin], 1.0f);
}

// Launcher for histc. Output must be zero-initialized. Input assumed contiguous float32 CUDA tensor.
extern "C" void histc_forward_launcher(
    const float* input,
    float* output,
    int num_elements,
    int bins,
    float min,
    float max,
    cudaStream_t stream
) {
    if (num_elements == 0 || bins <= 0) return;
    const int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    histc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, num_elements, bins, min, max);
}
