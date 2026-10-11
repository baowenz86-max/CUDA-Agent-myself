#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void conv_transpose1d_custom_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in, int C_out, int L_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream);

at::Tensor conv_transpose1d_custom_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (C_out)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight first dimension must match input channels");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must match output channels");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "computed output length must be positive");

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_custom_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(L_in),
        static_cast<int>(C_out),
        static_cast<int>(L_out),
        static_cast<int>(K),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream);

    return output;
}

void register_conv_transpose1d_custom(pybind11::module& m) {
    m.def("conv_transpose1d_custom", &conv_transpose1d_custom_forward,
          "Custom ConvTranspose1d forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
}

REGISTER_BINDING(conv_transpose1d_custom, register_conv_transpose1d_custom);
