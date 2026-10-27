// Custom ops binding for silu and conv_transpose1d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void silu_forward_launcher(
    float* out,
    const float* in,
    int64_t n,
    cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t length_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    cudaStream_t stream);

// SiLU wrapper
static torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "silu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "silu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "silu_forward: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    silu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream);

    return output;
}

// ConvTranspose1d wrapper
static torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Validate x
    TORCH_CHECK(x.is_cuda(), "conv_transpose1d_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "conv_transpose1d_forward: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "conv_transpose1d_forward: x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "conv_transpose1d_forward: x must be 3D [batch, in_channels, length_in]");

    // Validate weight
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be 3D [in_channels, out_channels, kernel_size]");

    int64_t batch = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t length_in = x.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(in_channels == weight.size(0), "conv_transpose1d_forward: weight in_channels mismatch");

    // Validate optional bias
    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "conv_transpose1d_forward: bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
        TORCH_CHECK(bias_tensor.is_contiguous(), "conv_transpose1d_forward: bias must be contiguous");
        TORCH_CHECK(bias_tensor.dim() == 1, "conv_transpose1d_forward: bias must be 1D [out_channels]");
        TORCH_CHECK(bias_tensor.size(0) == out_channels, "conv_transpose1d_forward: bias size mismatch");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    // Compute output length
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "conv_transpose1d_forward: invalid output length");

    auto output = torch::empty({batch, out_channels, length_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        batch,
        in_channels,
        out_channels,
        length_in,
        kernel_size,
        stride,
        padding,
        output_padding,
        stream);

    return output;
}

// Registration
static void register_custom_ops(pybind11::module& m) {
    m.def("silu_forward", &silu_forward, "SiLU activation",
          py::arg("x"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward, "1D transposed convolution",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
