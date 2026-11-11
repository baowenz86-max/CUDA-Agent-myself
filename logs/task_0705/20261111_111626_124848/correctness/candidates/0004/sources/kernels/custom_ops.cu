#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// argmax_dim: 对输入 [batch, length, channels] 沿 dim 求最大值和索引
// 此处 dim 固定为 1 (由契约保证), 但代码按 dim 参数处理, 但仅支持 dim=1
// 输出 values 与 indices 形状为 [batch, channels]
// -----------------------------------------------------------------------------

__global__ void argmax_dim_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ indices,
    float* __restrict__ values,
    int batch,
    int length,
    int channels)
{
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_outputs = (int64_t)batch * channels;
    if (out_idx >= total_outputs) return;

    int b = out_idx / channels;
    int c = out_idx % channels;

    const float* in_ptr = input + ((int64_t)b * length + 0) * channels + c;
    float best_val = -INFINITY;
    int64_t best_idx = 0;

    // 使用 grid-stride 循环处理超长 length
    for (int i = 0; i < length; ++i) {
        float val = in_ptr[(int64_t)i * channels];
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    values[out_idx] = best_val;
    indices[out_idx] = best_idx;
}

extern "C" void argmax_dim_launcher(
    const float* input,
    int64_t* indices,
    float* values,
    int batch,
    int length,
    int channels,
    int dim,
    cudaStream_t stream)
{
    // 本任务只支持 dim=1，但根据参考调用 dim=1
    if (dim != 1) {
        // 不支持，但为了避免未定义，留空
    }
    int64_t total_outputs = (int64_t)batch * channels;
    if (total_outputs <= 0) return;

    int threads = 256;
    int64_t blocks64 = (total_outputs + threads - 1) / threads;
    int blocks = (blocks64 > INT32_MAX) ? INT32_MAX : (int)blocks64;

    argmax_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, indices, values, batch, length, channels);
}

// -----------------------------------------------------------------------------
// float_cast: int64 -> float32 逐元素转换
// -----------------------------------------------------------------------------

__global__ void float_cast_kernel(
    const int64_t* __restrict__ input,
    float* __restrict__ output,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (float)input[idx];
    }
}

extern "C" void float_cast_launcher(
    const int64_t* input,
    float* output,
    int64_t n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = (blocks64 > INT32_MAX) ? INT32_MAX : (int)blocks64;
    float_cast_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// -----------------------------------------------------------------------------
// batch_norm_eval: y = (x - mean) / sqrt(var + eps) * weight + bias
// x: [batch, channels], 参数: [channels]
// -----------------------------------------------------------------------------

__global__ void batch_norm_eval_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ output,
    int batch,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;

    int c = idx % channels;
    float mean = running_mean[c];
    float var = running_var[c];
    float w = weight[c];
    float b = bias[c];
    float inv_std = rsqrtf(var + eps);
    float val = (input[idx] - mean) * inv_std * w + b;
    output[idx] = val;
}

extern "C" void batch_norm_eval_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    float* output,
    int batch,
    int channels,
    cudaStream_t stream)
{
    int total = batch * channels;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batch_norm_eval_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, running_mean, running_var, (float)eps, output, batch, channels);
}
