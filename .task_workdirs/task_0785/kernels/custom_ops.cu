#include <cuda_runtime.h>
#include <math.h>

// Kernel for computing mean over dim=1 with keepdim=True
// Input shape: [batch, channels, height, width], contiguous, float32
// Output shape: [batch, 1, height, width]
__global__ void mean_dim1_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                int batch,
                                int channels,
                                int height,
                                int width) {
    int total_output = batch * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_output; i += stride) {
        int b = i / (height * width);
        int hw = i % (height * width);
        int h = hw / width;
        int w = hw % width;

        float sum = 0.0f;
        int base = b * channels * height * width + h * width + w;
        for (int c = 0; c < channels; ++c) {
            sum += input[base + c * height * width];
        }
        output[i] = sum / channels;
    }
}

// Kernel for GELU activation (default approximate='none')
// y = x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(float* __restrict__ output,
                            const float* __restrict__ input,
                            int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        float x = input[i];
        // Use erf and constant sqrt(2)/2 = 0.70710678f
        output[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

// Host launcher for custom_mean_dim1
// Expects contiguous input tensor of shape [batch, channels, height, width]
// Output initialized to zeros before launch? The kernel writes every element, so not strictly needed.
extern "C" void custom_mean_dim1_launcher(float* output,
                                           const float* input,
                                           int batch,
                                           int channels,
                                           int height,
                                           int width,
                                           cudaStream_t stream) {
    int total_output = batch * height * width;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;

    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, channels, height, width);
}

// Host launcher for custom_gelu
extern "C" void custom_gelu_launcher(float* output,
                                      const float* input,
                                      int numel,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    gelu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}
