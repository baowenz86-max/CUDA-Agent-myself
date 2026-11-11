// kernels/custom_ops.cu
// 纯 CUDA kernel 实现 relu_forward 和 batch_norm1d_eval_forward
// 不包含 torch/ATen/pybind，只提供 extern "C" launcher 供 binding.cpp 调用

#include <cuda_runtime.h>

// ReLU 逐元素 kernel
__global__ void relu_kernel(float* out, const float* in, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        float v = in[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
}

// BatchNorm1d eval kernel
// 输入形状 [N, C, L]，按通道归一化
__global__ void batch_norm1d_eval_kernel(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = N * C * L;

    for (int64_t i = idx; i < total; i += stride) {
        // 计算当前元素所属的通道索引
        int64_t c = (i / L) % C;

        float mean = running_mean[c];
        float var = running_var[c];
        float scale = weight[c] / sqrtf(var + eps);
        float shift = bias[c] - mean * scale;

        out[i] = x[i] * scale + shift;
    }
}

// 外部 C launcher：relu
// 输出张量已由 binding 层分配，这里只写入
// numel: 总元素数
// stream: 当前 CUDA stream
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {

    if (numel <= 0) return;

    int block_size = 256;
    // 网格数量限制在 65535 以内，但随 numel 调整；使用 grid-stride loop 处理更大数据
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    relu_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        output, input, numel);
}

// 外部 C launcher：batch_norm1d_eval
// output, x, weight, bias, running_mean, running_var 都是 GPU 指针
// N, C, L 为维度；eps 为 double，kernel 内转为 float 以匹配运行精度
extern "C" void batch_norm1d_eval_launcher(
    float* output,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t L,
    cudaStream_t stream) {

    int64_t total = N * C * L;
    if (total <= 0) return;

    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    // 将 eps 转为 float，避免 kernel 中隐式转换
    float eps_f = static_cast<float>(eps);

    batch_norm1d_eval_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        output, x, weight, bias, running_mean, running_var, eps_f, N, C, L);
}
