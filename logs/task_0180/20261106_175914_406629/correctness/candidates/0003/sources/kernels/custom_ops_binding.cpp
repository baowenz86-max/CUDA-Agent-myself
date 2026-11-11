#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// External launcher declarations from custom_ops.cu
// -----------------------------------------------------------------------------
extern "C" void conv_transpose1d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    const float* input, const float* bias, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream);

extern "C" void round_forward_launcher(
    const float* input, float* output, int total,
    int input_stride, int output_stride, cudaStream_t stream);

extern "C" void triu_forward_launcher(
    const float* input, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream);

// ----------------------------------------------------------------
// conv_transpose1d_forward
// ----------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be a CUDA tensor if provided");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "bias must be float32");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int batch_size = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int input_length = input_contig.size(2);

    int weight_in_channels = weight_contig.size(0);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    TORCH_CHECK(in_channels == weight_in_channels, "input channels and weight in_channels must match");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.contiguous().data_ptr<float>() : nullptr;

    conv_transpose1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_length, static_cast<int>(output_length),
        kernel_size, static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(output_padding),
        stream);

    return output;
}

// ----------------------------------------------------------------
// add_bias_forward
// ----------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    int batch_size = x_contig.size(0);
    int out_channels = x_contig.size(1);
    int length = x_contig.size(2);

    TORCH_CHECK(bias_contig.numel() == out_channels, "bias must have out_channels elements");

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_forward_launcher(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, length,
        out_channels * length, length,
        out_channels * length, length,
        stream);

    return output;
}

// ----------------------------------------------------------------
// round_forward
// ----------------------------------------------------------------
torch::Tensor round_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int total = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    round_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        1, 1,
        stream);

    return output;
}

// ----------------------------------------------------------------
// triu_forward
// ----------------------------------------------------------------
torch::Tensor triu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    auto x_contig = x.contiguous();
    int batch_size = x_contig.size(0);
    int out_channels = x_contig.size(1);
    int length = x_contig.size(2);

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, length,
        out_channels * length, length,
        out_channels * length, length,
        stream);

    return output;
}

// ----------------------------------------------------------------
// Registration
// ----------------------------------------------------------------
void register_custom_ops(py::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward (custom CUDA kernel)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias with broadcasting",
          py::arg("x"), py::arg("bias"));
    m.def("round_forward", &round_forward,
          "Round elements (away from zero)",
          py::arg("x"));
    m.def("triu_forward", &triu_forward,
          "Upper triangular for last two dims",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
