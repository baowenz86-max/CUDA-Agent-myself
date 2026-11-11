#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../binding_registry.h"

namespace py = pybind11;

// 声明 .cu 中的 launcher
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
);

extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
);

// 内部实现：参数为 int64_t，符合契约
static torch::Tensor maxpool2d_forward_impl(
    torch::Tensor input,
    int64_t kernel_size_h,
    int64_t kernel_size_w,
    int64_t stride,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = (H + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions");

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size_h, kernel_size_w,
        stride, stride,
        padding_h, padding_w,
        dilation_h, dilation_w,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

// 绑定函数：接收 py::object 处理 tuple kernel_size
static torch::Tensor maxpool2d_forward(
    torch::Tensor input,
    py::object kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // 从 py::object 提取 kernel_size 的 int 值
    int64_t kernel_h, kernel_w;
    if (py::isinstance<py::tuple>(kernel_size)) {
        auto ks_tuple = kernel_size.cast<py::tuple>();
        TORCH_CHECK(ks_tuple.size() == 2, "kernel_size tuple must have 2 elements");
        kernel_h = ks_tuple[0].cast<int64_t>();
        kernel_w = ks_tuple[1].cast<int64_t>();
    } else if (py::isinstance<py::int_>(kernel_size)) {
        int64_t k = kernel_size.cast<int64_t>();
        kernel_h = k;
        kernel_w = k;
    } else {
        TORCH_CHECK(false, "kernel_size must be int or tuple of 2 ints");
    }

    // padding 可能是 tuple 或 int，这里 model_new.py 传入 padding[0]（int），但为兼容也处理 tuple
    int64_t padding_h, padding_w;
    if (py::isinstance<py::tuple>(py::cast(padding))) {
        auto pad_tuple = py::cast<py::tuple>(py::cast(padding));
        TORCH_CHECK(pad_tuple.size() == 2, "padding tuple must have 2 elements");
        padding_h = pad_tuple[0].cast<int64_t>();
        padding_w = pad_tuple[1].cast<int64_t>();
    } else {
        padding_h = padding;
        padding_w = padding;
    }

    // dilation 类似
    int64_t dilation_h, dilation_w;
    if (py::isinstance<py::tuple>(py::cast(dilation))) {
        auto dil_tuple = py::cast<py::tuple>(py::cast(dilation));
        TORCH_CHECK(dil_tuple.size() == 2, "dilation tuple must have 2 elements");
        dilation_h = dil_tuple[0].cast<int64_t>();
        dilation_w = dil_tuple[1].cast<int64_t>();
    } else {
        dilation_h = dilation;
        dilation_w = dilation;
    }

    return maxpool2d_forward_impl(
        input,
        kernel_h, kernel_w,
        stride,
        padding_h, padding_w,
        dilation_h, dilation_w
    );
}

// relu 绑定函数
static torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int64_t n = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "MaxPool2d forward",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"));
    m.def("relu_forward", &relu_forward,
          "ReLU forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
