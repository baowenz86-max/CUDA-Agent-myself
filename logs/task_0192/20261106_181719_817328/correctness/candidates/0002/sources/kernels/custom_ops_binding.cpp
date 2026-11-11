#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void bias_add_broadcast_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride,
    cudaStream_t stream);

extern "C" void std_batch_width_unbiased_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride,
    cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_width,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

// ConvTranspose1d forward
// This wrapper handles tensor checks, shape computation, allocation, and launcher call.
 torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }

    auto input_sizes = input.sizes();
    TORCH_CHECK(input_sizes.size() == 3, "input must be 3D");
    int64_t batch = input_sizes[0];
    int64_t in_channels = input_sizes[1];
    int64_t in_width = input_sizes[2];

    auto weight_sizes = weight.sizes();
    TORCH_CHECK(weight_sizes.size() == 3, "weight must be 3D");
    TORCH_CHECK(weight_sizes[0] == in_channels, "weight in_channels mismatch");
    int64_t out_channels = weight_sizes[1];
    int64_t kernel_size = weight_sizes[2];

    int64_t out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_width > 0, "output width must be positive");

    auto output = torch::empty({batch, out_channels, out_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(in_width),
        static_cast<int>(out_width),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream);

    return output;
}

// Bias add broadcast
torch::Tensor bias_add_broadcast(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    auto input_sizes = input.sizes();
    TORCH_CHECK(input_sizes.size() == 3, "input must be 3D");
    int64_t batch = input_sizes[0];
    int64_t out_channels = input_sizes[1];
    int64_t out_width = input_sizes[2];

    auto bias_sizes = bias.sizes();
    TORCH_CHECK(bias_sizes.size() == 2, "bias must be 2D");
    TORCH_CHECK(bias_sizes[0] == out_channels, "bias channel mismatch");
    TORCH_CHECK(bias_sizes[1] == 1, "bias second dim must be 1");

    auto input_contig = input.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_broadcast_forward_launcher(
        input_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(out_channels),
        static_cast<int>(out_width),
        static_cast<long long>(input_contig.stride(0)),
        static_cast<long long>(input_contig.stride(1)),
        static_cast<long long>(input_contig.stride(2)),
        stream);

    return output;
}

// Std batch width unbiased
torch::Tensor std_batch_width_unbiased(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    auto input_sizes = input.sizes();
    int64_t batch = input_sizes[0];
    int64_t out_channels = input_sizes[1];
    int64_t out_width = input_sizes[2];

    auto input_contig = input.contiguous();

    auto output = torch::empty({out_channels}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std_batch_width_unbiased_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(out_channels),
        static_cast<int>(out_width),
        static_cast<long long>(input_contig.stride(0)),
        static_cast<long long>(input_contig.stride(1)),
        static_cast<long long>(input_contig.stride(2)),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("bias_add_broadcast", &bias_add_broadcast,
          "Broadcast add bias [C,1] to [B,C,W]",
          py::arg("input"), py::arg("bias"));
    m.def("std_batch_width_unbiased", &std_batch_width_unbiased,
          "Unbiased std over batch and width dimensions",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
