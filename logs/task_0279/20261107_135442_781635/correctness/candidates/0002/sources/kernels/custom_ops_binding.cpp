#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Remove cudnn.h include and all cuDNN code; we'll use custom CUDA kernels.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

// Forward declarations of launchers from custom_ops.cu
extern "C" void normalize_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    cudaStream_t stream);

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int groups,
    cudaStream_t stream);

// Helper to get current CUDA stream
cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ConvTranspose1d implementation using custom kernel
torch::Tensor conv_transpose1d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Input checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_FLOAT32(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    }
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t kernel_size = weight.size(2);
    int64_t C_out = weight.size(1);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");
    TORCH_CHECK(weight.size(0) == C_in, "weight in_channels mismatch");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == C_out, "bias size mismatch with out_channels");
    }

    auto output = torch::empty({N, C_out, L_out}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().contiguous().data_ptr<float>();
    }

    cudaStream_t stream = getCurrentStream();

    conv_transpose1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream);

    return output;
}

// Normalize implementation using launcher from .cu
torch::Tensor normalize(torch::Tensor input, double p, int64_t dim) {
    CHECK_CUDA(input);
    CHECK_FLOAT32(input);
    TORCH_CHECK(input.dim() == 3, "normalize input must be 3D (N, C, L)");
    TORCH_CHECK(dim == 1, "normalize only supports dim=1");
    TORCH_CHECK(p == 2.0, "normalize only supports p=2");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = getCurrentStream();

    normalize_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(L),
        stream);

    return output;
}

// Conv1d implementation using custom kernel
torch::Tensor conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t groups) {
    // Input checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_FLOAT32(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    }
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels/groups, kernel_size)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t kernel_size = weight.size(2);
    int64_t C_out = weight.size(0);
    int64_t L_out = (L_in + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(L_out > 0, "output length must be positive");
    TORCH_CHECK(weight.size(1) * groups == C_in, "weight in_channels/groups mismatch");
    TORCH_CHECK(C_out % groups == 0, "out_channels must be divisible by groups");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == C_out, "bias size mismatch with out_channels");
    }

    auto output = torch::empty({N, C_out, L_out}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().contiguous().data_ptr<float>();
    }

    cudaStream_t stream = getCurrentStream();

    conv1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(groups),
        stream);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d", &conv_transpose1d, "ConvTranspose1d via custom kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("normalize", &normalize, "L2 normalization along dim=1",
          pybind11::arg("input"), pybind11::arg("p"), pybind11::arg("dim"));
    m.def("conv1d", &conv1d, "Conv1d via custom kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("groups"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
