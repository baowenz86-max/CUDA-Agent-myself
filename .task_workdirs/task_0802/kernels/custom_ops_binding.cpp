#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明从 custom_ops.cu 导出的 launcher
extern "C" void custom_softmax_channel_only_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream);

extern "C" void custom_prelu_launcher(
    const float* x,
    float* y,
    const float* weight,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream);

// 验证输入张量并获取 stride，返回是否成功及 strides
static bool get_4d_strides(const torch::Tensor& x, long long& stride_n, long long& stride_c, long long& stride_h, long long& stride_w) {
    if (x.dim() != 4) {
        throw std::invalid_argument("Input tensor must be 4-dimensional (N, C, H, W)");
    }
    stride_n = x.stride(0);
    stride_c = x.stride(1);
    stride_h = x.stride(2);
    stride_w = x.stride(3);
    return true;
}

torch::Tensor custom_softmax_channel_only(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    long long stride_n, stride_c, stride_h, stride_w;
    get_4d_strides(x, stride_n, stride_c, stride_h, stride_w);

    // 输出张量，保持与输入相同的形状、dtype、device、layout
    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_channel_only_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream
    );
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_channel_only kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

torch::Tensor custom_prelu(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(x.dim() == 4, "Input x must be 4-dimensional");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1-dimensional");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(weight.size(0) == C, "Weight size must match input channel dimension");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    long long stride_n, stride_c, stride_h, stride_w;
    get_4d_strides(x, stride_n, stride_c, stride_h, stride_w);

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_prelu_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_prelu kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_channel_only", &custom_softmax_channel_only,
          "Softmax over channel dimension (dim=1) for 4D tensor",
          pybind11::arg("x"));
    m.def("custom_prelu", &custom_prelu,
          "PReLU activation with per-channel weight",
          pybind11::arg("x"), pybind11::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
