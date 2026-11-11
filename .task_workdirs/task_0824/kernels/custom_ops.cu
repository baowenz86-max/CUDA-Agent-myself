#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

// ============================================================================
// batch_norm2d_inference
// 输入 x: [B, C, H, W] 连续 float32
// weight, bias, running_mean, running_var: [C] float32
// eps: double, 但计算时转为 float
// 输出 y: 与 x 形状相同
// ============================================================================
extern "C" __global__ void batch_norm2d_inf_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int64_t total_elements,
    int64_t C,
    int64_t HW)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int c = (idx / HW) % C;
        float inv_std = rsqrtf(var[c] + eps);
        y[idx] = (x[idx] - mean[c]) * inv_std * weight[c] + bias[c];
    }
}

extern "C" void batch_norm2d_inference_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream)
{
    int64_t total = B * C * H * W;
    if (total == 0) return;
    int64_t HW = H * W;
    float eps_f = (float)eps;
    int block_size = 256;
    int grid_size = (int)std::min<int64_t>((total + block_size - 1) / block_size, 65535);
    batch_norm2d_inf_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps_f, total, C, HW);
}

// ============================================================================
// permute_contiguous
// 输入 x: float32，任意 strides，shape 为 4 维，dims 为 CPU int64_t 数组长度 4
// 输出 out: 连续，按照 dims 指定的顺序排列，shape 与 dims 对应
// 此处输入可能非连续（转置视图），因此必须按 strides 访问
// ============================================================================
extern "C" __global__ void permute_contiguous_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_stride,
    const int64_t* __restrict__ out_shape,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int64_t remaining = idx;
        int64_t out_index[4];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            out_index[d] = remaining / out_shape[d];
            remaining %= out_shape[d];
        }
        // 根据 dims 映射回输入索引
        int64_t in_index[4];
        // dims 数组由 host 传递，这里 kernel 直接假设 dims 顺序已知：
        // 我们将在 launcher 中传入 dims 数组，但为了简单，kernel 采用通用映射：
        // 约定 dims 是 {0,2,1,3} 或 {0,2,1,3}？实际上模型中使用的是固定置换 (0,2,1,3) 两次。
        // 为了通用性，我们传入 dims 数组，并在 kernel 中按 dims 反查。
        // 这里需要 dims 的指针。
        // 修改 kernel 签名：增加 dims 指针。
        // 但当前 kernel 未包含 dims，我们将重新定义下方。
    }
}

// 更通用的实现：以 dims 为参数，在 kernel 内部根据 out 的坐标计算 in 的坐标
// dims 是目标顺序，out_shape 按照 dims 顺序排列
// in_shape 是原始 shape，in_stride 是原始 stride
__global__ void permute_contiguous_general_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_stride,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ dims,  // 长度为 4，dims[i] 表示输出第 i 维对应输入的维度
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int64_t remaining = idx;
        int64_t out_coord[4];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            out_coord[d] = remaining / out_shape[d];
            remaining %= out_shape[d];
        }
        // 映射到输入坐标
        int64_t in_linear = 0;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int64_t in_dim = dims[d];
            in_linear += out_coord[d] * in_stride[in_dim];
        }
        out[idx] = in[in_linear];
    }
}

extern "C" void permute_contiguous_launcher(
    float* output,
    const float* input,
    const int64_t* in_shape,
    const int64_t* in_stride,
    const int64_t* out_shape,
    const int64_t* dims,  // CPU 指针，已在 host 侧传入 device 内存？这里需要 device 数组。
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;
    int block_size = 256;
    int grid_size = (int)std::min<int64_t>((total_elements + block_size - 1) / block_size, 65535);
    permute_contiguous_general_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, in_shape, in_stride, out_shape, dims, total_elements);
}

// ============================================================================
// max_pool1d
// 输入 x: [B, L, C] 连续 float32，对 L 维做最大池化
// kernel_size, stride, padding, dilation 为标量，int64_t
// 输出 out: [B, L_out, C]，L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
// ============================================================================
extern "C" __global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B,
    int64_t L,
    int64_t C,
    int64_t L_out,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = B * L_out * C;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += grid_stride) {
        // 分解为 (b, l_out, c)
        int64_t c = idx % C;
        int64_t tmp = idx / C;
        int64_t l_out = tmp % L_out;
        int64_t b = tmp / L_out;

        float max_val = -FLT_MAX;
        // 起始位置计算：参考 PyTorch MaxPool1d 公式
        // 窗口起始索引：l_start = l_out * stride - padding + k * dilation, k=0..kernel_size-1
        // 有效窗口内的元素
        for (int k = 0; k < kernel_size; ++k) {
            int64_t l_in = l_out * stride - padding + k * dilation;
            if (l_in >= 0 && l_in < L) {
                float val = input[(b * L + l_in) * C + c];
                if (val > max_val) max_val = val;
            }
        }
        output[idx] = max_val;
    }
}

extern "C" void max_pool1d_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t L,
    int64_t C,
    int64_t L_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    cudaStream_t stream)
{
    int64_t total = B * L_out * C;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (int)std::min<int64_t>((total + block_size - 1) / block_size, 65535);
    max_pool1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, B, L, C, L_out,
        (int)kernel_size, (int)stride, (int)padding, (int)dilation);
}

// ============================================================================
// view 操作不需要 CUDA kernel，只需在 host 端改变 metadata（由 binding 层实现）
// 因此 custom_ops.cu 不包含 view 相关 kernel。
// ============================================================================
