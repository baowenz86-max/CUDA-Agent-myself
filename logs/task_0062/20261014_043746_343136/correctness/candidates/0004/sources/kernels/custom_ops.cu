#include <cuda_runtime.h>

// Sigmoid kernel: applies 1/(1+exp(-x)) elementwise.
// Grid-stride loop for arbitrary numel.
__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        // Stable sigmoid: 1 / (1 + exp(-val))
        output[i] = 1.0f / (1.0f + __expf(-val));
    }
}

// ConvTranspose1d kernel: computes output[n,k,l] = sum_{c,s} input[n,c,l+s] * weight[c,k,s]
// where input is [N, C, L], weight is [C, K, kernel_size], output is [N, K, L_out].
// L_out = (L - 1) * stride - 2 * padding + kernel_size + output_padding.
// The kernel assumes input, weight, output are contiguous float32 arrays.
// To compute each output element, we iterate over c and kernel positions that contribute.
// We use a grid-stride loop over output elements; each thread computes one output element.
// The indexing: output index = ((n*K + k)*L_out + l), we need to find corresponding input and weight indices.
// The relation: output[n,k,l] = sum_c sum_s input[n,c, l+s] * weight[c,k,s] for valid (l+s) after accounting for stride and padding.
// Actually for ConvTranspose1d: output[n, k, l] = sum_c sum_s input[n, c, floor((l + padding - s)/stride)] * weight[c, k, s] if (l + padding - s) % stride == 0 and index in [0, L-1].
// Here l ranges [0, L_out-1], s ranges [0, kernel_size-1].
__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int64_t N, int64_t C, int64_t K,
                                        int64_t L, int64_t L_out,
                                        int kernel_size,
                                        int stride, int padding) {
    int64_t total_outputs = N * K * L_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_outputs; i += stride_grid) {
        int64_t l = i % L_out;
        int64_t k = (i / L_out) % K;
        int64_t n = i / (L_out * K);
        float acc = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            for (int s = 0; s < kernel_size; ++s) {
                int64_t t = l + padding - s;
                if (t < 0 || (t % stride) != 0) continue;
                int64_t l_in = t / stride;
                if (l_in >= 0 && l_in < L) {
                    // input index: n*C*L + c*L + l_in
                    // weight index: c*K*kernel_size + k*kernel_size + s
                    acc += input[(n * C + c) * L + l_in] * weight[(c * K + k) * kernel_size + s];
                }
            }
        }
        output[i] = acc;
    }
}

// Bias addition kernel: output[n,k,l] += bias[k]
__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int64_t N, int64_t K, int64_t L_out) {
    int64_t total_outputs = N * K * L_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_outputs; i += stride_grid) {
        int64_t k = (i / L_out) % K;
        output[i] += bias[k];
    }
}

// Launcher for sigmoid_forward. Assumes input is float32 contiguous CUDA tensor.
// output_ptr must already be allocated (by binding) with same shape.
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    // Cap blocks to a reasonable maximum to avoid too many blocks (grid-stride will handle remainder).
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for conv_transpose1d_forward.
// input: [N, C, L]
// weight: [C, K, kernel_size]
// output: [N, K, L_out]
// bias: [K] or nullptr
// stride, padding, output_padding (output_padding not used in kernel, only for length calc)
extern "C" void conv_transpose1d_forward_launcher(const float* input,
                                                  const float* weight,
                                                  const float* bias,
                                                  float* output,
                                                  int64_t N, int64_t C, int64_t K,
                                                  int64_t L, int64_t L_out,
                                                  int kernel_size,
                                                  int stride, int padding,
                                                  cudaStream_t stream) {
    if (L_out <= 0) return;
    const int threads = 256;
    int64_t total_outputs = N * K * L_out;
    int blocks = static_cast<int>((total_outputs + threads - 1) / threads);
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, N, C, K, L, L_out, kernel_size, stride, padding);
    if (bias != nullptr) {
        // Bias addition kernel
        add_bias_kernel<<<blocks, threads, 0, stream>>>(output, bias, N, K, L_out);
    }
}
