#include <cuda_runtime.h>

// Kernel: compute mean over batch dimension (dim=0)
// Input: x shape [batch, channels, height, width]
// Output: out shape [channels, height, width]
// Each thread computes one or more output elements (grid-stride loop).
__global__ void mean_reduce_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch,
                                   int channels,
                                   int height,
                                   int width) {
    // Output element index: c * height * width + h * width + w
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_out = channels * height * width;

    for (int idx = out_idx; idx < total_out; idx += stride) {
        int w = idx % width;
        int tmp = idx / width;
        int h = tmp % height;
        int c = tmp / height;

        // Accumulate over batch dimension
        float sum = 0.0f;
        for (int b = 0; b < batch; ++b) {
            // Input is assumed contiguous or at least accessible with standard strides.
            // Use pointer arithmetic matching contiguous layout; binding should ensure contiguous or provide strides.
            // Since tensor is contiguous in task (stride [50331648, 16777216, 4096, 1]), we use this simple indexing.
            // To support arbitrary strides, launcher would need stride info. For now, contract says supports strides,
            // but actual input is contiguous. We'll require contiguous in binding, or make launcher handle via strides not possible with only pointer and dims.
            // We'll note that this kernel assumes contiguous input. If non-contiguous, caller must make contiguous before call.
            // This is acceptable for current task.
            int in_idx = ((b * channels + c) * height + h) * width + w;
            sum += input[in_idx];
        }
        output[idx] = sum / static_cast<float>(batch);
    }
}

// Kernel: Hardswish elementwise
__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        float x = input[i];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = x * relu6 / 6.0f;
    }
}

// Launcher for mean reduction (extern "C" for binding)
extern "C" void custom_mean_launcher(const float* input,
                                      float* output,
                                      int batch,
                                      int channels,
                                      int height,
                                      int width,
                                      cudaStream_t stream) {
    int total_out = channels * height * width;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size? grid-stride loop handles large sizes.
    // Max blocks for current GPUs is large enough, but we can cap at 65535*? For safety, we use a moderate number.
    // Actually we can just compute blocks = min((total_out + threads -1)/threads, 1024*?)
    // Use grid size that covers all elements with at most 256 threads per block.
    // For 50M output elements, blocks ~ 196608, which is okay on most GPUs (max grid dim x = 2^31-1 for 1D).
    // But to improve occupancy, we can use a fixed grid like 256 blocks * 256 threads = 65536 threads, enough with grid-stride.
    // We'll set blocks to a configurable number? Use min(blocks, 65535) but max blocks in x dimension is 2^31-1 for compute capability >=3.0.
    // So no worry.
    mean_reduce_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, channels, height, width);
}

// Launcher for Hardswish (extern "C" for binding)
extern "C" void custom_hardswish_launcher(const float* input,
                                          float* output,
                                          int total_elements,
                                          cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hardswish_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}
