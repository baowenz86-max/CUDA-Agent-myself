#include <cuda_runtime.h>
#include <cstdint>

// Kernel: mean over channels (dim=1) with keepdim=True, then fmod by divisor.
// Input: x shape [B, C, H, W], contiguous in memory, float32.
// Output: out shape [B, 1, H, W], contiguous, float32.
// Each thread block handles one (b,h,w) position and loops over channels.
__global__ void mean_channel_fmod_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int batch,
                                         int channels,
                                         int height,
                                         int width,
                                         float divisor) {
    // Flattened index over output elements: index = b*height*width + h*width + w
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch * height * width;
    if (out_idx >= total_out) return;

    int w = out_idx % width;
    int tmp = out_idx / width;
    int h = tmp % height;
    int b = tmp / height;

    const float* in_base = input + ((long long)b * channels + 0LL) * height * width; // base for channel 0
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        // Access contiguous along width; for each channel, the offset is c * H * W + h * W + w
        const float* in_ptr = input + ((long long)b * channels + c) * (long long)height * width + (long long)h * width + w;
        sum += *in_ptr;
    }
    float mean = sum / (float)channels;
    // fmod with divisor
    output[out_idx] = fmodf(mean, divisor);
}

// Launcher for mean_channel_fmod. Assumes input and output are contiguous CUDA memory.
extern "C" void mean_channel_fmod_launcher(const float* input,
                                           float* output,
                                           int batch,
                                           int channels,
                                           int height,
                                           int width,
                                           float divisor,
                                           cudaStream_t stream) {
    if (!input || !output || batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        // In production binding would handle error, but here we can return silently after checking.
        // No error throwing allowed; launcher only returns. Binding should ensure validity.
        return;
    }
    int total_out = batch * height * width;
    // Choose block size 256 and compute grid size with grid-stride? Since each thread handles one output, we can use enough blocks.
    // For large B*H*W (here 512*512*512 = 134,217,728 elements), 256 threads per block => 524,288 blocks, well within limits.
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    mean_channel_fmod_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, channels, height, width, divisor);
}

// Kernel: elementwise fmod scalar.
__global__ void fmod_scalar_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t N,
                                   float modulus) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        output[i] = fmodf(input[i], modulus);
    }
}

// Launcher for fmod_scalar. Assumes contiguous CUDA memory.
extern "C" void fmod_scalar_launcher(const float* input,
                                     float* output,
                                     size_t N,
                                     float modulus,
                                     cudaStream_t stream) {
    if (!input || !output || N == 0) {
        return;
    }
    int threads = 256;
    // Calculate grid size to cover N, but use grid-stride loop for safety.
    // We can set grid to min((N + threads - 1)/threads, some cap like 65535*32) but grid-stride loop handles any.
    size_t blocks_required = (N + threads - 1) / threads;
    // Cap grid at 2^31-1 blocks; if more, we rely on grid-stride loop with grid dims limited.
    // But actual grid limit is 2^31-1 for x dimension; we can set grid size to min(blocks_required, 65535) and use stride.
    // For very large N (134M), blocks_required ~ 524k, which is fine. So we can just use blocks_required, no need for capping.
    int blocks = (int)blocks_required;
    fmod_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, N, modulus);
}
