#include <cuda_runtime.h>
#include <math.h>

// ConvTranspose1d kernel: gather method
// output[b, oc, l] = bias[oc] + sum_{ic, k} input[b, ic, i] * weight[ic, oc, k]
// where i = (l + padding - k) / stride, valid when (l + padding - k) % stride == 0 and i in [0, L_in)
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int L_in,
    int C_out, int K, int stride, int padding, int output_padding,
    int L_out)
{
    int total = B * C_out * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int t = idx; t < total; t += step) {
        int l = t % L_out;
        int oc = (t / L_out) % C_out;
        int b = t / (L_out * C_out);

        float acc = (bias != nullptr) ? bias[oc] : 0.0f;
        for (int k = 0; k < K; ++k) {
            int x = l + padding - k;
            if (x < 0 || (x % stride) != 0) continue;
            int i = x / stride;
            if (i < 0 || i >= L_in) continue;
            for (int ic = 0; ic < C_in; ++ic) {
                acc += input[((b * C_in + ic) * L_in) + i] * weight[((ic * C_out + oc) * K) + k];
            }
        }
        output[t] = acc;
    }
}

__global__ void rsqrt_eps_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float eps,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += step) {
        float val = input[i];
        output[i] = rsqrtf(fabsf(val) + eps);
    }
}

// Softmax over last dimension: input shape (outer, inner)
__global__ void softmax_lastdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int inner)
{
    for (int row = blockIdx.x; row < outer; row += gridDim.x) {
        const float* row_in = input + (size_t)row * inner;
        float* row_out = output + (size_t)row * inner;

        // find max
        float max_val = row_in[0];
        for (int i = threadIdx.x; i < inner; i += blockDim.x) {
            float v = row_in[i];
            if (v > max_val) max_val = v;
        }
        __shared__ float block_max[1];
        if (threadIdx.x == 0) block_max[0] = max_val;
        __syncthreads();
        max_val = block_max[0];
        // sum of exp
        __shared__ float block_sum[1];
        float sum = 0.0f;
        for (int i = threadIdx.x; i < inner; i += blockDim.x) {
            sum += __expf(row_in[i] - max_val);
        }
        if (threadIdx.x == 0) block_sum[0] = sum;
        __syncthreads();
        sum = block_sum[0];
        // normalize
        for (int i = threadIdx.x; i < inner; i += blockDim.x) {
            row_out[i] = __expf(row_in[i] - max_val) / sum;
        }
    }
}

// Correct block-wide max reduction for softmax
__device__ void blockReduceMax(float& val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }
    val = smem[0];
}

__device__ void blockReduceSum(float& val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    val = smem[0];
}

// Softmax using block reductions (assumes inner <= blockDim)
__global__ void softmax_lastdim_kernel_block(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int inner)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= outer) return;

    const float* row_in = input + (size_t)row * inner;
    float* row_out = output + (size_t)row * inner;

    // load row into shared memory
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        smem[i] = row_in[i];
    }
    __syncthreads();

    // find max
    float max_val = (threadIdx.x < inner) ? smem[threadIdx.x] : -INFINITY;
    blockReduceMax(max_val, smem + inner); // use second half for reduction
    __syncthreads();

    // compute sum of exp
    float sum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        sum += __expf(smem[i] - max_val);
    }
    blockReduceSum(sum, smem + inner);
    __syncthreads();

    // write output
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        row_out[i] = __expf(smem[i] - max_val) / sum;
    }
}

// Launcher for conv_transpose1d
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in,
    int C_out, int K, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    int total = B * C_out * L_out;

    // Zero initialize output (important for accumulate if kernel uses atomics, but gather kernel overwrites, still zero to be safe)
    cudaMemsetAsync(output, 0, (size_t)total * sizeof(float), stream);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid max, kernel uses grid-stride
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        B, C_in, L_in, C_out, K, stride, padding, output_padding, L_out);
}

// Launcher for rsqrt_eps
extern "C" void rsqrt_eps_launcher(
    const float* input,
    float* output,
    float eps,
    int total,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    rsqrt_eps_kernel<<<blocks, threads, 0, stream>>>(input, output, eps, total);
}

// Launcher for softmax last dim
extern "C" void softmax_lastdim_launcher(
    const float* input,
    float* output,
    int outer,
    int inner,
    cudaStream_t stream)
{
    if (inner <= 1024) {
        int threads = inner;
        // round up to warp multiple
        threads = (threads + 31) / 32 * 32;
        if (threads < 32) threads = 32;
        int smem_size = (inner + threads) * sizeof(float);
        softmax_lastdim_kernel_block<<<outer, threads, smem_size, stream>>>(input, output, outer, inner);
    } else {
        // fallback to simple per-thread reduction (not optimal)
        int threads = 256;
        int blocks = outer;
        softmax_lastdim_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, inner);
    }
}
