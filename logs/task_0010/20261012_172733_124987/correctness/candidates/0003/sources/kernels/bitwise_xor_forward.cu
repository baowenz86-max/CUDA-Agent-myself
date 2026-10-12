#include <cuda_runtime.h>

__global__ void bitwise_xor_kernel(unsigned char* output, const unsigned char* input, const unsigned char* bias,
                                   int channels, int height, int width, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // Compute channel index for bias broadcast
    int channel = (idx / (height * width)) % channels;
    output[idx] = input[idx] ^ bias[channel];
}

extern "C" void bitwise_xor_launcher(
    unsigned char* output,
    const unsigned char* input,
    const unsigned char* bias,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bitwise_xor_kernel<<<blocks, threads, 0, stream>>>(output, input, bias, channels, height, width, total);
}
