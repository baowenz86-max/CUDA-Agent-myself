#include <cuda_runtime.h>
#include <math.h>

// Helper macro for CUDA error checking (not used in launchers to keep them simple)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while (0)

// ============ 1. ConvTranspose1d ============
// Input: (N, C_in, L_in), Weight: (C_in, C_out, K), Bias: (C_out,)
// Output: (N, C_out, L_out) with L_out = (L_in-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1
// Here dilation=1, groups=1
// We assume contiguous tensors.

template <int BLOCK_SIZE>
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding)
{
    // Each thread computes one output element: (n, oc, l)
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N * C_out * L_out) return;

    int l = idx % L_out;
    int tmp = idx / L_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float acc = (bias != nullptr) ? bias[oc] : 0.0f;

    // Iterate over input positions and kernel offsets
    // Formula: for each k in [0, K-1], for each ic, if (l + padding - k) % stride == 0
    // and input position = (l + padding - k) / stride is within [0, L_in-1], accumulate.
    for (int ic = 0; ic < C_in; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int x = l + padding - k;
            if (x >= 0 && x % stride == 0) {
                int i = x / stride;
                if (i >= 0 && i < L_in) {
                    acc += input[(n * C_in + ic) * L_in + i] * weight[(ic * C_out + oc) * kernel_size + k];
                }
            }
        }
    }
    output[idx] = acc;
}

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total_elements = N * C_out * L_out;
    const int block_size = 256;
    // Since N*C_out*L_out can be up to 33.5M, and grid dimension x max is 2^31-1,
    // but we use 1D grid, so no issue.
    int grid_size = (total_elements + block_size - 1) / block_size;
    conv_transpose1d_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, output_padding);
}

// ============ 2. Rsqrt of abs(x) + eps ============
__global__ void rsqrt_eps_kernel(const float* __restrict__ input, float* __restrict__ output, int numel, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float v = fabsf(input[idx]) + eps;
    output[idx] = rsqrtf(v);
}

extern "C" void rsqrt_eps_launcher(const float* input, float* output, int numel, float eps, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    rsqrt_eps_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel, eps);
}

// ============ 3. Softmax along last dimension ============
// Each block handles one row (one sample along last dim).
// For each (n, c), softmax over L_out elements.
// We use one block per row, with blockDim.x threads reducing.

__global__ void softmax_lastdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size, int inner_size)
{
    __shared__ float s_max;
    __shared__ float s_sum;

    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Input pointer offset for this row
    const float* row_in = input + outer_idx * inner_size;
    float* row_out = output + outer_idx * inner_size;

    // Find max
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float v = row_in[i];
        if (v > thread_max) thread_max = v;
    }
    // Reduce max across threads in block
    // We'll use shared memory and simple reduction.
    __shared__ float sdata[256]; // assume blockDim.x <= 256
    int tid = threadIdx.x;
    sdata[tid] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = tid; i < inner_size; i += blockDim.x) {
        thread_sum += expf(row_in[i] - max_val);
    }
    // Reduce sum
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

    // Write softmax output
    for (int i = tid; i < inner_size; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_val) / sum_val;
    }
}

extern "C" void softmax_lastdim_launcher(
    const float* input,
    float* output,
    int outer_size, int inner_size,
    cudaStream_t stream)
{
    // Each block handles one row.
    // outer_size can be up to N * C_out = 128*16 = 2048, which is fine.
    const int block_size = 256;
    // If inner_size is large, one block with 256 threads is enough,
    // but we should handle if inner_size > 65535? Actually grid dimension x is max 65535 for blocks, but outer_size=2048 < 65535, okay.
    // However, if outer_size > 65535, we'd need a 2D grid or loop over rows. Not needed here.
    int grid_size = outer_size;
    softmax_lastdim_kernel<<<grid_size, block_size, 0, stream>>>(input, output, outer_size, inner_size);
}
