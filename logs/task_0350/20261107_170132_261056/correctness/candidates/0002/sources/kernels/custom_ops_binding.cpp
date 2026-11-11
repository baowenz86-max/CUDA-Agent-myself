// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream);

extern "C" void conv1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

// Helper: get current CUDA stream
static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ConvTranspose1d forward using custom kernel
at::Tensor conv_transpose1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D: [in_channels, out_channels, kernel_size]");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t in_length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "output length must be positive");

    auto output = at::empty({batch, out_channels, out_length}, input_contig.options());

    cudaStream_t stream = get_stream();

    const float* bias_ptr = nullptr;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == at::kFloat, "bias must be CUDA float32");
        TORCH_CHECK(bias.numel() == out_channels, "bias size must match out_channels");
        bias_ptr = bias.contiguous().data_ptr<float>();
    }

    conv_transpose1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        (int)batch,
        (int)in_channels,
        (int)out_channels,
        (int)in_length,
        (int)out_length,
        (int)kernel_size,
        (int)stride,
        (int)padding,
        (int)output_padding,
        stream
    );

    return output;
}

// Conv1d forward using custom kernel
at::Tensor conv1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D: [in_channels, out_channels, kernel_size]");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t in_length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(out_length > 0, "output length must be positive");

    auto output = at::empty({batch, out_channels, out_length}, input_contig.options());

    cudaStream_t stream = get_stream();

    const float* bias_ptr = nullptr;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == at::kFloat, "bias must be CUDA float32");
        TORCH_CHECK(bias.numel() == out_channels, "bias size must match out_channels");
        bias_ptr = bias.contiguous().data_ptr<float>();
    }

    conv1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        (int)batch,
        (int)in_channels,
        (int)out_channels,
        (int)in_length,
        (int)out_length,
        (int)kernel_size,
        (int)stride,
        (int)padding,
        stream
    );

    return output;
}

// ReLU forward
at::Tensor relu_forward(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    auto input_contig = input.contiguous();
    auto output = at::empty_like(input_contig);
    cudaStream_t stream = get_stream();
    long long numel = input_contig.numel();
    relu_forward_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    return output;
}

// MaxPool1d forward
at::Tensor max_pool1d_forward(
    at::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    auto input_contig = input.contiguous();
    int64_t batch = input_contig.size(0);
    int64_t channels = input_contig.size(1);
    int64_t length_in = input_contig.size(2);
    int64_t length_out = (length_in + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(length_out > 0, "output length must be positive");
    auto output = at::empty({batch, channels, length_out}, input_contig.options());
    cudaStream_t stream = get_stream();
    max_pool1d_forward_launcher(
        input_contig.data_ptr<float>(), output.data_ptr<float>(),
        (int)batch, (int)channels, (int)length_in, (int)length_out,
        (int)kernel_size, (int)stride, (int)padding, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using custom CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride") = 2, pybind11::arg("padding") = 1, pybind11::arg("output_padding") = 1);
    m.def("conv1d_forward", &conv1d_forward,
          "Conv1d forward using custom CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride") = 1, pybind11::arg("padding") = 1);
    m.def("relu_forward", &relu_forward, "ReLU forward");
    m.def("max_pool1d_forward", &max_pool1d_forward, "MaxPool1d forward",
          pybind11::arg("input"), pybind11::arg("kernel_size") = 2,
          pybind11::arg("stride") = 2, pybind11::arg("padding") = 0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
