#include <cuda_runtime.h>

// LeakyReLU kernel: output = input > 0 ? input : negative_slope * input
__global__ void leaky_relu_kernel(float* output, const float* input, long long numel, float negative_slope) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : negative_slope * val;
    }
}

// InstanceNorm kernel for 2D spatial normalization
// Each block processes one (N,C) pair: computes mean and variance over H*W, then normalizes.
// Input layout: input pointer with strides: n_stride, c_stride, h_stride, w_stride
// Output layout: same as input (output pointer passed in, assumed same layout as input)
// weight and bias are optional, length C
__global__ void instance_norm_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    float eps)
{
    int nc = blockIdx.x; // 0..N*C-1
    int n = nc / C;
    int c = nc % C;
    long long n_offset = (long long)n * n_stride;
    long long c_offset = (long long)c * c_stride;
    const float* in_base = input + n_offset + c_offset;
    float* out_base = output + n_offset + c_offset;

    int HW = H * W;
    float sum = 0.0f;
    float sumsq = 0.0f;

    // Compute mean and variance using thread reduction
    extern __shared__ float sdata[];
    // sdata[0..blockDim.x-1] for sum, sdata[blockDim.x..2*blockDim.x-1] for sumsq
    float* s_sum = sdata;
    float* s_sumsq = sdata + blockDim.x;

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int idx = tid; idx < HW; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        float val = in_base[h * h_stride + w * w_stride];
        local_sum += val;
        local_sumsq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sumsq = s_sumsq[0];
    float mean = total_sum / HW;
    float var = total_sumsq / HW - mean * mean; // biased variance
    float inv_std = rsqrtf(var + eps);

    float gamma = (weight != nullptr) ? weight[c] : 1.0f;
    float beta = (bias != nullptr) ? bias[c] : 0.0f;

    // Normalize and write output
    for (int idx = tid; idx < HW; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        float val = in_base[h * h_stride + w * w_stride];
        float normalized = (val - mean) * inv_std;
        out_base[h * h_stride + w * w_stride] = normalized * gamma + beta;
    }
}

// Launcher for leaky_relu_forward
extern "C" void leaky_relu_forward_launcher(
    float* output,
    const float* input,
    long long numel,
    float negative_slope,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    leaky_relu_kernel<<<(int)grid_size, block_size, 0, stream>>>(output, input, numel, negative_slope);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handled in binding via throw? We can't throw here without torch. Just ignore? 
        // We'll rely on binding to check after. But launcher should not cause undefined behavior.
    }
}

// Launcher for instance_norm_forward
extern "C" void instance_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    float eps,
    cudaStream_t stream)
{
    int nc_total = N * C;
    // Choose block size (threads per (N,C) block)
    int block_size = 256;
    if (H * W < 256) block_size = 128;
    if (H * W < 128) block_size = 64;
    if (H * W < 64) block_size = 32;

    int shared_mem_size = 2 * block_size * sizeof(float); // for sum and sumsq arrays
    // Ensure block_size <= max threads (1024)
    if (block_size > 1024) block_size = 1024;

    instance_norm_kernel<<<nc_total, block_size, shared_mem_size, stream>>>(
        output, input, weight, bias, N, C, H, W,
        n_stride, c_stride, h_stride, w_stride, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Same as above
    }
}
