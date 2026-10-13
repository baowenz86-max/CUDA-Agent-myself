#include <cuda_runtime.h>

// Custom CUDA kernel for ConvTranspose1d forward
__global__ void conv_transpose1d_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int L_in, int L_out, int kernel_size,
    int stride, int padding, int output_padding)
{
    // Each thread computes one output element at position (n, co, l_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;
    
    int l_out = idx % L_out;
    int co = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);
    
    float sum = (bias != nullptr) ? bias[co] : 0.0f;
    
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < kernel_size; ++k) {
            int l_in_index = l_out + padding - k;
            if (l_in_index % stride != 0) continue;
            l_in_index /= stride;
            if (l_in_index < 0 || l_in_index >= L_in) continue;
            sum += input[n * C_in * L_in + ci * L_in + l_in_index] *
                   weight[ci * C_out * kernel_size + co * kernel_size + k];
        }
    }
    
    output[idx] = sum;
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void conv_transpose1d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int L_in, int L_out, int kernel_size,
    int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, C_out, L_in, L_out, kernel_size,
        stride, padding, output_padding);
}
