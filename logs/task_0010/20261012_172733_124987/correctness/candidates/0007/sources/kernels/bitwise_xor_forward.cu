// bitwise_xor_forward.cu
// Kernel for bitwise XOR between uint8 tensor and broadcasted uint8 bias.
// The bias is provided as a float32 tensor (or any dtype) and cast to uint8 on the fly.
// Input tensor is uint8, output is uint8.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel: each thread computes one element.
// grid: total number of elements / block_size, with grid-stride loop for safety.
// Arguments:
// - out: output uint8 array (N elements)
// - in: input uint8 array (N elements)
// - bias: bias array, length = C (number of channels). It is passed as float32 pointer.
// - C: number of channels (input's second dimension)
// - spatial_size: number of elements per channel (H*W)
// - N: total number of elements (C*spatial_size)
__global__ void bitwise_xor_kernel(
    unsigned char* __restrict__ out,
    const unsigned char* __restrict__ in,
    const float* __restrict__ bias,
    int C,
    int spatial_size,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        int c = (i / spatial_size) % C; // channel index, assumes layout N,C,H,W contiguous
        unsigned char b = static_cast<unsigned char>(static_cast<int>(bias[c]));
        out[i] = in[i] ^ b;
    }
}

// Launcher function (extern "C" so it can be called from binding)
// Parameters:
// - out: output uint8 pointer
// - in: input uint8 pointer
// - bias: bias float32 pointer (or any 4-byte type, but we interpret as float)
// - C: number of channels
// - spatial_size: H*W
// - N: total elements
// - stream: CUDA stream
extern "C" void bitwise_xor_launcher(
    unsigned char* out,
    const unsigned char* in,
    const float* bias,
    int C,
    int spatial_size,
    int N,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    bitwise_xor_kernel<<<blocks, threads, 0, stream>>>(
        out, in, bias, C, spatial_size, N);
}
