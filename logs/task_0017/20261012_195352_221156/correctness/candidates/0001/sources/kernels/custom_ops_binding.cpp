#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_custom_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void min_scalar_custom_launcher(
    const float* input,
    float* output,
    int numel,
    float scalar,
    cudaStream_t stream);

extern "C" void erf_custom_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream);

// ConvTranspose1d wrapper
torch::Tensor conv_transpose1d_custom(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "conv_transpose1d_custom: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_custom: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "conv_transpose1d_custom: bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "conv_transpose1d_custom: input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_custom: weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(bias.dim() == 1, "conv_transpose1d_custom: bias must be 1D (C_out)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "conv_transpose1d_custom: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_custom: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose1d_custom: bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "conv_transpose1d_custom: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_custom: weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "conv_transpose1d_custom: bias must be contiguous");

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int C_out = weight.size(1);
    int K = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "conv_transpose1d_custom: weight input channels mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "conv_transpose1d_custom: bias output channels mismatch");
    TORCH_CHECK(stride > 0, "conv_transpose1d_custom: stride must be positive");
    TORCH_CHECK(padding >= 0, "conv_transpose1d_custom: padding must be non-negative");
    TORCH_CHECK(output_padding >= 0 && output_padding < stride, "conv_transpose1d_custom: output_padding must be in [0, stride)");

    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "conv_transpose1d_custom: computed output length is non-positive");

    auto output = torch::empty({N, C_out, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_custom_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out,
        K, static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(output_padding),
        stream);

    return output;
}

// Min with scalar wrapper
torch::Tensor min_scalar_custom(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "min_scalar_custom: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "min_scalar_custom: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "min_scalar_custom: input must be contiguous");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    min_scalar_custom_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        static_cast<float>(scalar),
        stream);
    return output;
}

// Erf wrapper
torch::Tensor erf_custom(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "erf_custom: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "erf_custom: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "erf_custom: input must be contiguous");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    erf_custom_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream);
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_custom", &conv_transpose1d_custom,
          "Custom ConvTranspose1d forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("min_scalar_custom", &min_scalar_custom,
          "Custom element-wise min with scalar (CUDA)",
          py::arg("input"), py::arg("scalar"));
    m.def("erf_custom", &erf_custom,
          "Custom element-wise erf (CUDA)",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
