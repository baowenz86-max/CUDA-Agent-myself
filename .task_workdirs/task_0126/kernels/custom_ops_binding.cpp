#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(
    const float* input, float* output, int numel, int C, int L,
    int stride_b, int stride_c, int stride_l, cudaStream_t stream);

extern "C" void maxpool1d_forward_launcher(
    const float* input, float* output, int B, int C, int L, int out_len,
    int kernel_size, int stride, int padding, int dilation,
    int stride_b, int stride_c, int stride_l, cudaStream_t stream);

// GELU wrapper
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "gelu_forward: input must be 3D [B, C, L]");

    auto B = x.size(0);
    auto C = x.size(1);
    auto L = x.size(2);

    // Get strides, but if non-contiguous and strides not supported, we could make contiguous.
    // The .cu kernel handles arbitrary strides, so pass them directly.
    int stride_b = x.stride(0);
    int stride_c = x.stride(1);
    int stride_l = x.stride(2);

    // Output same shape as input, contiguous
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(x.numel()),
        static_cast<int>(C),
        static_cast<int>(L),
        stride_b,
        stride_c,
        stride_l,
        stream);

    return output;
}

// MaxPool1d wrapper
torch::Tensor maxpool1d_forward(
    torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "maxpool1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "maxpool1d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "maxpool1d_forward: input must be 3D [B, C, L]");
    TORCH_CHECK(kernel_size > 0, "maxpool1d_forward: kernel_size must be positive");
    TORCH_CHECK(stride > 0, "maxpool1d_forward: stride must be positive");
    TORCH_CHECK(padding >= 0, "maxpool1d_forward: padding must be non-negative");
    TORCH_CHECK(dilation > 0, "maxpool1d_forward: dilation must be positive");

    auto B = x.size(0);
    auto C = x.size(1);
    auto L = x.size(2);

    // Compute output length using PyTorch MaxPool1d formula (floor, ceil_mode=false)
    int64_t out_len = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_len > 0, "maxpool1d_forward: computed output length is non-positive");

    // Input strides (kernel handles arbitrary strides)
    int stride_b = x.stride(0);
    int stride_c = x.stride(1);
    int stride_l = x.stride(2);

    // Allocate output contiguous with shape [B, C, out_len]
    auto output = torch::empty({B, C, out_len}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool1d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(out_len),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        stride_b,
        stride_c,
        stride_l,
        stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU exact forward", py::arg("x"));
    m.def("maxpool1d_forward", &maxpool1d_forward, "MaxPool1d forward",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
