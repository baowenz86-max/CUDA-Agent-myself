#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

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

static torch::Tensor maxpool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions");

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        dilation, dilation,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

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

void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "MaxPool2d forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("dilation"));
    m.def("relu_forward", &relu_forward,
          "ReLU forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
