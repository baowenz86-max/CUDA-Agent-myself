#include <cuda_runtime.h>
#include <stdint.h>

// ReflectionPad2d for uint8 tensors
// Grid-stride loop over output elements
__global__ void reflection_pad2d_uint8_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int padding)
{
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        // Decompose linear index into (n, c, i_out, j_out)
        int64_t j_out = linear % W_out;
        int64_t tmp = linear / W_out;
        int64_t i_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Reflect indices for input coordinates
        int64_t i_in = i_out - padding;
        int64_t j_in = j_out - padding;

        // Reflect function: reflect(t, M) = min(abs(t), 2*M-2-abs(t))
        // t is in range [-padding, M-1+padding]
        int64_t H_in_minus_1 = H_in - 1;
        int64_t W_in_minus_1 = W_in - 1;

        int64_t abs_i = i_in >= 0 ? i_in : -i_in;
        int64_t reflect_i = abs_i;
        int64_t two_H_minus_2 = 2 * H_in_minus_1;
        int64_t reflect_candidate = two_H_minus_2 - abs_i;
        reflect_i = reflect_i < reflect_candidate ? reflect_i : reflect_candidate;

        int64_t abs_j = j_in >= 0 ? j_in : -j_in;
        int64_t reflect_j = abs_j;
        int64_t two_W_minus_2 = 2 * W_in_minus_1;
        reflect_candidate = two_W_minus_2 - abs_j;
        reflect_j = reflect_j < reflect_candidate ? reflect_j : reflect_candidate;

        // Input index: ((n*C + c)*H_in + reflect_i)*W_in + reflect_j
        int64_t input_idx = ((n * C + c) * H_in + reflect_i) * W_in + reflect_j;
        output[linear] = input[input_idx];
    }
}

// Bitwise XOR for uint8 tensors
template <typename T>
__global__ void bitwise_xor_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t total)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        output[i] = a[i] ^ b[i];
    }
}

// Launcher for reflection_pad2d_uint8
extern "C" void reflection_pad2d_uint8_launcher(
    const uint8_t* input,
    uint8_t* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int padding,
    cudaStream_t stream)
{
    int64_t total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = min((total + block_size - 1) / block_size, (int64_t)65535);
    reflection_pad2d_uint8_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, padding);
}

// Launcher for bitwise_xor_uint8
extern "C" void bitwise_xor_uint8_launcher(
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* output,
    int64_t total,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = min((total + block_size - 1) / block_size, (int64_t)65535);
    bitwise_xor_kernel<uint8_t><<<grid_size, block_size, 0, stream>>>(
        a, b, output, total);
}
