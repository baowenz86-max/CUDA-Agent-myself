#include <cuda_runtime.h>
#include <math.h>

// ============ 1. ConvTranspose1d (scatter with atomics) ============
// Input: (N, C_in, L_in), Weight: (C_in, C_out, K), Bias: (C_out,)
// Output: (N, C_out, L_out) with L_out = (L_in-1)*stride - 2*padding + kernel_size + output_padding
// We use one thread per input element per kernel element per output channel?
// Simpler: one thread per input element (n, ic, i) and kernel element k, output channel oc.
// This can be many threads: N*C_in*L_in*K*C_out. For our case: 128*3*8192*3*16 = ~452M threads, too many.
// Better: use one block per output element with loop over contributions? The direct output approach is simpler but may have incorrect indexing.
// But we suspect the direct approach had a bug; if we switch to scatter, we can use atomicAdd.
// We'll do a grid-stride loop over input elements and for each, loop over oc and k, scatter to output.
// This gives total work = N*C_in*L_in*K*C_out, which is still manageable with grid-stride loop.
__global__ void conv_transpose1d_scatter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding)
{
    int total = N * C_in * L_in;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;
    for (int flat = idx; flat < total; flat += stride_total) {
        int i = flat % L_in;
        int tmp = flat / L_in;
        int ic = tmp % C_in;
        int n = tmp / C_in;

        float in_val = input[flat];

        for (int oc = 0; oc < C_out; ++oc) {
            for (int k = 0; k < kernel_size; ++k) {
                int l = i * stride - padding + k + output_padding;
                if (l >= 0 && l < L_out) {
                    float w = weight[(ic * C_out + oc) * kernel_size + k];
                    atomicAdd(&output[(n * C_out + oc) * L_out + l], in_val * w);
                }
            }
        }
    }

    // Add bias: we need to add bias to each output element. We can do it in a separate kernel or in this one after sync.
    // Better: after all atomics, we need to add bias. We'll launch a separate kernel for bias or do it here if we ensure all threads done? We can use a grid sync? Not safe. Launch separate kernel.
}

__global__ void add_bias_kernel(float* output, const float* bias, int N, int C_out, int L_out)
{
    int total = N * C_out * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int l = idx % L_out;
        int tmp = idx / L_out;
        int oc = tmp % C_out;
        output[idx] += bias[oc];
    }
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
    // Zero initialize output
    cudaMemsetAsync(output, 0, sizeof(float) * N * C_out * L_out, stream);

    int total = N * C_in * L_in;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    conv_transpose1d_scatter_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, output_padding);

    if (bias != nullptr) {
        int total_out = N * C_out * L_out;
        int grid_bias = (total_out + block_size - 1) / block_size;
        add_bias_kernel<<<grid_bias, block_size, 0, stream>>>(output, bias, N, C_out, L_out);
    }
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
__global__ void softmax_lastdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size, int inner_size)
{
    __shared__ float s_max;
    __shared__ float s_sum;

    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row_in = input + outer_idx * inner_size;
    float* row_out = output + outer_idx * inner_size;

    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float v = row_in[i];
        if (v > thread_max) thread_max = v;
    }

    __shared__ float sdata[256];
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

    float thread_sum = 0.0f;
    for (int i = tid; i < inner_size; i += blockDim.x) {
        thread_sum += expf(row_in[i] - max_val);
    }

    sdata[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

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
    const int block_size = 256;
    int grid_size = outer_size;
    softmax_lastdim_kernel<<<grid_size, block_size, 0, stream>>>(input, output, outer_size, inner_size);
}
