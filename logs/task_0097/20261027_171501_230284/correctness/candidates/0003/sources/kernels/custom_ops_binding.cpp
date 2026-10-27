#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_histc_launcher(
    const float* input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream);

extern "C" void custom_conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

namespace {

inline void checkCuda(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(error));
    }
}

// Custom ConvTranspose1d forward implementation
// Input: [batch, in_channels, length], weight: [in_channels, out_channels, kernel_size], bias: [out_channels]
// stride, padding, output_padding scalar
// Uses pure CUDA kernel launcher, no cuDNN to avoid missing headers.
torch::Tensor custom_conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    TORCH_CHECK(bias.size(0) == weight.size(1), "bias size must match out_channels");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight in_channels must match input channels");

    // Make contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    // Compute output length: (length - 1)*stride - 2*padding + kernel_size + output_padding
    int64_t output_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive");

    // Create output tensor
    auto output = torch::empty({batch, out_channels, output_length}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(length),
        static_cast<int>(output_length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    checkCuda(cudaGetLastError(), "conv_transpose1d launch");

    return output;
}

// Custom histc forward
// Input: contiguous CUDA float tensor, bins, min, max
// Output: tensor of shape [bins] with counts (float)
torch::Tensor custom_histc_forward(
    torch::Tensor input,
    int64_t bins,
    double min_val,
    double max_val)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_histc_launcher(
        input.data_ptr<float>(),
        input.numel(),
        static_cast<int>(bins),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        output.data_ptr<float>(),
        stream
    );

    checkCuda(cudaGetLastError(), "histc launch");

    return output;
}

} // anonymous namespace

void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d_forward,
          "Custom ConvTranspose1d",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 2, py::arg("padding") = 1, py::arg("output_padding") = 1);
    m.def("custom_histc", &custom_histc_forward,
          "Custom histogram",
          py::arg("input"), py::arg("bins") = 100,
          py::arg("min") = 0.0, py::arg("max") = 100.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
