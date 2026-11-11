#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 custom_ops.cu 中的 launcher
extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

extern "C" void mean_dim_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream);

// PyTorch 包装函数：常数填充
torch::Tensor pad_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0, "Padding must be non-negative");

    int64_t H_out = H + pad_top + pad_bottom;
    int64_t W_out = W + pad_left + pad_right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pad_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream);

    // 可选：检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "pad_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch 包装函数：沿 dim=1 求均值
torch::Tensor mean_dim_forward(torch::Tensor x, int64_t dim) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N,C,H,W)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for mean");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({N, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("pad_forward", &pad_forward,
          "Constant padding forward",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
    m.def("mean_dim_forward", &mean_dim_forward,
          "Mean along dimension forward",
          py::arg("x"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
