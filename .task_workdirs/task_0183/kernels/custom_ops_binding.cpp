// Bindings for reflection_pad2d and conv2d.
// This file contains only PyTorch tensor handling and CUDA launcher calls,
// no cuDNN dependency.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

#include "../binding_registry.h"

// Error checking macro for CUDA
#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      const char* err_str = cudaGetErrorString(status);                        \
      throw std::runtime_error(std::string("CUDA error: ") + err_str);         \
    }                                                                          \
  } while (0)

// extern "C" launcher declarations from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int pad,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream);

extern "C" void conv2d_direct_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int P, int Q,
    cudaStream_t stream);

// ----------------------------------------------------------------------------
// ReflectionPad2d forward
// ----------------------------------------------------------------------------
torch::Tensor reflection_pad2d(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "reflection_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "reflection_pad2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "reflection_pad2d: input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(padding >= 0, "reflection_pad2d: padding must be >= 0");

    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_h = in_h + 2 * padding;
    int out_w = in_w + 2 * padding;

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    long long in_stride_b = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);

    long long out_stride_b = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        (int)padding,
        in_stride_b,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        stream
    );

    CHECK_CUDA(cudaGetLastError());

    return output;
}

// ----------------------------------------------------------------------------
// Conv2d forward using direct CUDA kernel (no cuDNN)
// ----------------------------------------------------------------------------
torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "conv2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "conv2d: input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(weight.is_cuda(), "conv2d: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d: weight must be float32");
    TORCH_CHECK(weight.dim() == 4, "conv2d: weight must be 4D (out_channels, in_channels, kh, kw)");
    TORCH_CHECK(stride > 0, "conv2d: stride must be positive");
    TORCH_CHECK(padding >= 0, "conv2d: padding must be >= 0");

    torch::Tensor bias;
    bool has_bias = false;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "conv2d: bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv2d: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "conv2d: bias must be 1D (out_channels)");
        TORCH_CHECK(bias.size(0) == weight.size(0), "conv2d: bias size must match out_channels");
        has_bias = true;
    }

    // Ensure contiguous for direct kernel
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    int K = weight_contig.size(0);
    int R = weight_contig.size(2);
    int S = weight_contig.size(3);

    int pad_h = padding;
    int pad_w = padding;
    int stride_h = stride;
    int stride_w = stride;

    int P = (H + 2 * pad_h - R) / stride_h + 1;
    int Q = (W + 2 * pad_w - S) / stride_w + 1;
    TORCH_CHECK(P > 0 && Q > 0, "conv2d: invalid output dimensions");

    auto output = torch::empty({N, K, P, Q}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

    conv2d_direct_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K, R, S,
        pad_h, pad_w,
        stride_h, stride_w,
        P, Q,
        stream
    );

    CHECK_CUDA(cudaGetLastError());

    return output;
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d", &reflection_pad2d,
          "ReflectionPad2d forward (CUDA)",
          py::arg("input"),
          py::arg("padding"));

    m.def("conv2d", &conv2d,
          "Conv2d forward using direct CUDA kernel (no cuDNN)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
