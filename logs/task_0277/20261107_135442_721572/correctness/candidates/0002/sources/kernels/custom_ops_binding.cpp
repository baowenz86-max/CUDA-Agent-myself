#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void selu_forward_launcher(
    float* out,
    const float* in,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void conv2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int K, int C_per_group, int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream
);

namespace {

// 辅助函数：检查 CUDA 错误
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

} // anonymous namespace

// selu_forward：对输入张量逐元素应用 SELU
torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // 处理非连续输入：SELU kernel 支持任意 stride，但为简单和性能，如果有必要则连续化
    // 注意：如果输入本身就是连续的，则 direct 使用，否则创建一个连续副本
    torch::Tensor x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    checkCudaError("selu_forward_launcher failed");

    return output;
}

// conv2d_forward：使用自定义 CUDA kernel 执行 2D 卷积
torch::Tensor conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(x.dim() == 4, "Input x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (out_channels, in_channels/groups, kH, kW)");

    // 尺寸信息
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t K = weight.size(0); // out_channels
    int64_t C_per_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    TORCH_CHECK(groups >= 1, "groups must be positive");
    TORCH_CHECK(C == C_per_group * groups, "Input channels must equal C_per_group * groups");
    TORCH_CHECK(K % groups == 0, "Output channels must be divisible by groups");

    // 计算输出尺寸（floor 公式）
    int64_t H_out = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = torch::empty({N, K, H_out, W_out}, x.options());

    // 连续化输入和权重
    torch::Tensor x_contig = x.contiguous();
    torch::Tensor weight_contig = weight.contiguous();

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 确定 bias 指针
    float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == K, "Bias shape must be [out_channels]");
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    // 调用卷积 launcher
    conv2d_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        (int)N, (int)C, (int)H, (int)W,
        (int)K, (int)C_per_group, (int)kH, (int)kW,
        (int)stride, (int)stride,
        (int)padding, (int)padding,
        (int)dilation, (int)dilation,
        (int)groups,
        (int)H_out, (int)W_out,
        stream
    );

    checkCudaError("conv2d_forward_launcher failed");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("selu_forward", &selu_forward, "SELU activation forward",
          pybind11::arg("x"));
    m.def("conv2d_forward", &conv2d_forward, "2D convolution forward using custom CUDA kernel",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("dilation") = 1,
          pybind11::arg("groups") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
