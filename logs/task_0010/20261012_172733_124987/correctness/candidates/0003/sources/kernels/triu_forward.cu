#include <cuda_runtime.h>

__global__ void triu_kernel(float* output, const float* input, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;

    int w = idx % width;
    int h = (idx / width) % height;
    output[idx] = (w >= h) ? input[idx] : 0.0f;
}

extern "C" void triu_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    triu_kernel<<<blocks, threads, 0, stream>>>(output, input, batch, channels, height, width);
}
