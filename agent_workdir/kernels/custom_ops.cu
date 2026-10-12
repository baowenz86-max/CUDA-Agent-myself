#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// 1. log10_lt kernel: out[i] = (log10(x[i]) < threshold) ? 1.0f : 0.0f
// -----------------------------------------------------------------------------
__global__ void log10_lt_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                int64_t numel,
                                float threshold) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        // log10f is fast and accurate enough; threshold comparison as in PyTorch
        output[idx] = (__log10f(val) < threshold) ? 1.0f : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 2. bias_add kernel: output[n,c,l] = input[n,c,l] + bias[c]  (bias shape: (C,1))
// -----------------------------------------------------------------------------
__global__ void bias_add_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                const float* __restrict__ bias,
                                int64_t N,
                                int64_t C,
                                int64_t L,
                                int64_t total_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int64_t l = idx % L;
        int64_t tmp = idx / L;
        int64_t c = tmp % C;
        // bias index is [c] (flattened, since bias shape is (C,1))
        output[idx] = input[idx] + bias[c];
    }
}

// -----------------------------------------------------------------------------
// 3. ConvTranspose1d kernel (batch, channels, length)
//    Input:  (N, C_in, L_in)
//    Weight: (C_in, C_out/groups, kernel_size)
//    Output: (N, C_out, L_out)
//    stride, padding, output_padding (only 0 or 1 for stride=2 to keep simple)
//    Each thread computes one output element.
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int N, int C_in, int L_in,
                                        int C_out, int L_out,
                                        int kernel_size,
                                        int stride, int padding,
                                        int output_padding, int groups) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C_out * L_out;
    if (out_idx >= total_out) return;

    int n = out_idx / (C_out * L_out);
    int rem = out_idx % (C_out * L_out);
    int c_out = rem / L_out;
    int l_out = rem % L_out;

    int group = c_out / (C_out / groups);
    int cin_per_group = C_in / groups;
    int cout_per_group = C_out / groups;
    int c_in_start = group * cin_per_group;

    float acc = 0.0f;

    for (int k = 0; k < kernel_size; ++k) {
        // Compute corresponding input index: l_in = (l_out + padding - k) / stride
        int tmp = l_out + padding - k;
        if (tmp % stride != 0) continue;
        int l_in = tmp / stride;
        if (l_in < 0 || l_in >= L_in) continue;

        // Loop over input channels in this group
        for (int cin_local = 0; cin_local < cin_per_group; ++cin_local) {
            int c_in = c_in_start + cin_local;
            int c_out_in_group = c_out - group * cout_per_group;
            float w = weight[cin_local + c_in * cout_per_group * kernel_size + c_out_in_group * kernel_size + k];
            float x = input[(n * C_in + c_in) * L_in + l_in];
            acc += x * w;
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    output[out_idx] = acc;
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return; \
        } \
    } while (0)

// launcher for log10_lt
// Arguments: output, input, numel, threshold, stream
extern "C" void log10_lt_launcher(float* output,
                                  const float* input,
                                  int64_t numel,
                                  float threshold,
                                  cudaStream_t stream) {
    const int block = 256;
    int64_t grid64 = (numel + block - 1) / block;
    if (grid64 > 2147483647LL) grid64 = 2147483647LL; // cap to int max
    int grid = (int)grid64;
    log10_lt_kernel<<<grid, block, 0, stream>>>(output, input, numel, threshold);
}

// launcher for bias_add
// Arguments: output, input, bias, dims N,C,L, stream
extern "C" void bias_add_launcher(float* output,
                                  const float* input,
                                  const float* bias,
                                  int64_t N,
                                  int64_t C,
                                  int64_t L,
                                  cudaStream_t stream) {
    int64_t total_elements = N * C * L;
    const int block = 256;
    int64_t grid64 = (total_elements + block - 1) / block;
    if (grid64 > 2147483647LL) grid64 = 2147483647LL;
    int grid = (int)grid64;
    bias_add_kernel<<<grid, block, 0, stream>>>(output, input, bias, N, C, L, total_elements);
}

// launcher for conv_transpose1d
// Arguments: input, weight, bias, output, dims, conv params, stream
extern "C" void conv_transpose1d_launcher(const float* input,
                                          const float* weight,
                                          const float* bias,
                                          float* output,
                                          int N, int C_in, int L_in,
                                          int C_out, int L_out,
                                          int kernel_size,
                                          int stride, int padding,
                                          int output_padding, int groups,
                                          cudaStream_t stream) {
    int total_out = N * C_out * L_out;
    const int block = 256;
    int grid = (total_out + block - 1) / block;
    conv_transpose1d_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, L_in, C_out, L_out,
        kernel_size, stride, padding, output_padding, groups);
}
