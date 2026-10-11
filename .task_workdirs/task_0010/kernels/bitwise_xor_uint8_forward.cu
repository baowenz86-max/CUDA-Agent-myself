#include <cuda_runtime.h>
#include <stdint.h>

// Kernel to compute bitwise XOR between two uint8 tensors with broadcasting.
// Contract: x has shape (N, C, H, W), bias has shape (C, 1, 1). So bias is broadcast along N, H, W.
// output[i] = x[i] ^ bias[c], where c = (i / (H*W)) % C.
__global__ void bitwise_xor_uint8_kernel(
    uint8_t* output,
    const uint8_t* x,
    const uint8_t* bias,
    int64_t num_elements,
    int64_t HW,
    int64_t C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Determine channel index c for this element
    int64_t c = (idx / HW) % C;
    output[idx] = x[idx] ^ bias[c];
}

// Launcher
extern "C" void bitwise_xor_uint8_forward_launcher(
    uint8_t* output,
    const uint8_t* x,
    const uint8_t* bias,
    int64_t num_elements,
    int64_t HW,
    int64_t C,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    bitwise_xor_uint8_kernel<<<blocks, threads, 0, stream>>>(
        output, x, bias, num_elements, HW, C);
}
