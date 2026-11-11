// kernels/custom_ops_binding.cpp
// PyTorch bindings for custom CUDA ops: batchnorm2d_forward, square_forward, avgpool2d_forward.
// This file contains only tensor handling, validation, launcher calls, and pybind registration.
// No CUDA kernels are implemented in this file.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare extern "C" launchers from custom_ops.cu
extern "C" void batchnorm2d_forward_cuda(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3,
    cudaStream_t stream);

extern "C" void square_forward_cuda(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3,
    cudaStream_t stream);

extern "C" void avgpool2d_forward_cuda(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    bool count_include_pad,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    cudaStream_t stream);

// Helper: create a tensor with the same sizes and strides as input (for elementwise ops)
torch::Tensor empty_like_same_strides(const torch::Tensor& x) {
    return torch::empty_strided(x.sizes(), x.strides(), x.options());
}

// ===================== batchnorm2d_forward =====================
torch::Tensor batchnorm2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Check CUDA
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    // Check dtype float32
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    // Check dimensions: x must be 4D [N, C, H, W]
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Check that parameter tensors are 1D with size C
    auto check_1d = [&](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.dim() == 1 && t.size(0) == C, name, " must be 1D with size C=", C);
    };
    check_1d(weight, "weight");
    check_1d(bias, "bias");
    check_1d(running_mean, "running_mean");
    check_1d(running_var, "running_var");

    // Optional: check contiguous for parameters
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // Get strides (in elements) as value
    int64_t stride_x0 = x.stride(0);
    int64_t stride_x1 = x.stride(1);
    int64_t stride_x2 = x.stride(2);
    int64_t stride_x3 = x.stride(3);

    // Allocate output with same strides as x
    torch::Tensor out = empty_like_same_strides(x);
    int64_t stride_out0 = out.stride(0);
    int64_t stride_out1 = out.stride(1);
    int64_t stride_out2 = out.stride(2);
    int64_t stride_out3 = out.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batchnorm2d_forward_cuda(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        out.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_out0, stride_out1, stride_out2, stride_out3,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batchnorm2d_forward kernel failed: ", cudaGetErrorString(err));

    return out;
}

// ===================== square_forward =====================
torch::Tensor square_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "square_forward currently only supports 4D tensors");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t stride_x0 = x.stride(0);
    int64_t stride_x1 = x.stride(1);
    int64_t stride_x2 = x.stride(2);
    int64_t stride_x3 = x.stride(3);

    torch::Tensor out = empty_like_same_strides(x);
    int64_t stride_out0 = out.stride(0);
    int64_t stride_out1 = out.stride(1);
    int64_t stride_out2 = out.stride(2);
    int64_t stride_out3 = out.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    square_forward_cuda(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_out0, stride_out1, stride_out2, stride_out3,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "square_forward kernel failed: ", cudaGetErrorString(err));

    return out;
}

// ===================== avgpool2d_forward =====================
torch::Tensor avgpool2d_forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool count_include_pad) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "Invalid pooling parameters: output dimension <= 0");

    int64_t stride_x0 = x.stride(0);
    int64_t stride_x1 = x.stride(1);
    int64_t stride_x2 = x.stride(2);
    int64_t stride_x3 = x.stride(3);

    torch::Tensor out = torch::empty({N, C, outH, outW}, x.options());
    TORCH_CHECK(out.is_contiguous(), "Internal error: output not contiguous");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avgpool2d_forward_cuda(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
        count_include_pad,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avgpool2d_forward kernel failed: ", cudaGetErrorString(err));

    return out;
}

// ===================== Registration =====================
void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm2d_forward", &batchnorm2d_forward,
          "BatchNorm2d forward (eval mode)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));  // required, no default
    m.def("square_forward", &square_forward,
          "Element-wise square",
          py::arg("x"));
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "AvgPool2d forward",
          py::arg("x"),
          py::arg("kernel_size"),       // required
          py::arg("stride"),             // required
          py::arg("padding"),            // required
          py::arg("count_include_pad") = true); // optional with default
}

REGISTER_BINDING(custom_ops, register_custom_ops);
