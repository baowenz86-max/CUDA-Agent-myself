#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
    cudaError_t flatten_launcher(
        const float* input,
        float* output,
        long long total_elements,
        cudaStream_t stream);

    cudaError_t conv_transpose1d_forward_launcher(
        const float* x,
        const float* weight,
        const float* bias,
        float* output,
        int batch,
        int in_channels,
        int out_channels,
        int length,
        int L_out,
        int kernel_size,
        int stride,
        int padding,
        int output_padding,
        cudaStream_t stream);
}

// PyTorch wrapper for ConvTranspose1d
at::Tensor conv_transpose1d_forward(
    at::Tensor x,
    at::Tensor weight,
    std::optional<at::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, in_channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [in_channels, out_channels, kernel_size]");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    // Bias check
    at::Tensor bias_tensor;
    bool has_bias = bias.has_value();
    if (has_bias) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D [out_channels]");
        TORCH_CHECK(bias_tensor.size(0) == out_channels, "bias size must match out_channels");
    }

    // Compute output length
    int L_out = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "L_out must be positive");

    // Allocate output
    auto output = at::empty({batch, out_channels, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    const float* bias_ptr = has_bias ? bias_tensor.data_ptr<float>() : nullptr;
    cudaError_t err = conv_transpose1d_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        length,
        L_out,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "conv_transpose1d_forward_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for flatten
at::Tensor flatten(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");

    long long total_elements = x.numel();
    auto output = at::empty({x.size(0), total_elements / x.size(0)}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = flatten_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "flatten_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("flatten", &flatten,
          "Flatten tensor to 2D (batch, -1)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);