#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int kernel_size, int stride, int padding, int dilation, bool ceil_mode,
    int64_t stride_n, int64_t stride_c, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    cudaStream_t stream);

torch::Tensor maxpool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    // Get dimensions from input tensor
    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H_in = static_cast<int>(input.size(2));
    int W_in = static_cast<int>(input.size(3));

    // Compute output dimensions (same formula as in launcher)
    int H_out, W_out;
    if (ceil_mode) {
        H_out = static_cast<int>((H_in + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1);
        W_out = static_cast<int>((W_in + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1);
    } else {
        H_out = static_cast<int>((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
        W_out = static_cast<int>((W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    }

    // Allocate output tensor with same dtype/device and appropriate size
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // Get input strides (int64_t)
    int64_t stride_n = input.stride(0);
    int64_t stride_c = input.stride(1);
    int64_t stride_h_in = input.stride(2);
    int64_t stride_w_in = input.stride(3);

    // Output strides are contiguous by construction, but we pass them for clarity
    int64_t stride_n_out = output.stride(0);
    int64_t stride_c_out = output.stride(1);
    int64_t stride_h_out = output.stride(2);
    int64_t stride_w_out = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    maxpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        ceil_mode,
        stride_n, stride_c, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "2D max pooling forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
