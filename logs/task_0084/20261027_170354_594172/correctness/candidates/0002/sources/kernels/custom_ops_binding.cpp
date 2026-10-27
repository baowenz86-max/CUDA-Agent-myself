#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launcher from custom_ops.cu
extern "C" void elu_add_forward_launcher(
    float* output,
    const float* input,
    const float* bias,
    int64_t total_elements,
    int C,
    int H,
    int W,
    int input_stride_n,
    int input_stride_c,
    int input_stride_h,
    int input_stride_w,
    int output_stride_n,
    int output_stride_c,
    int output_stride_h,
    int output_stride_w,
    float alpha,
    cudaStream_t stream
);

// Declare cuDNN convolution function from cudnn_conv.cpp
torch::Tensor cudnn_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups);

// cuDNN convolution forward with bias (wrapper)
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    return cudnn_conv2d_forward(input, weight, bias, stride, padding, dilation, groups);
}

// Fused ELU + bias add forward
torch::Tensor elu_add_forward(
    torch::Tensor input,
    torch::Tensor bias,
    double alpha) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D [C,1,1]");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK(bias.size(0) == C, "bias first dimension must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape [C,1,1]");

    // Get strides (may be negative? typically positive). Handle as int64_t, cast to int for kernel signature.
    int input_stride_n = (int)input.stride(0);
    int input_stride_c = (int)input.stride(1);
    int input_stride_h = (int)input.stride(2);
    int input_stride_w = (int)input.stride(3);

    // Allocate output with same shape and strides as input, contiguous by default
    auto output = torch::empty_like(input);

    int output_stride_n = (int)output.stride(0);
    int output_stride_c = (int)output.stride(1);
    int output_stride_h = (int)output.stride(2);
    int output_stride_w = (int)output.stride(3);

    int64_t total_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    elu_add_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        total_elements,
        (int)C,
        (int)H,
        (int)W,
        input_stride_n,
        input_stride_c,
        input_stride_h,
        input_stride_w,
        output_stride_n,
        output_stride_c,
        output_stride_h,
        output_stride_w,
        (float)alpha,
        stream
    );

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "2D convolution forward using cuDNN",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("groups") = 1);

    m.def("elu_add_forward", &elu_add_forward,
          "Fused bias add + ELU",
          py::arg("input"),
          py::arg("bias"),
          py::arg("alpha") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
