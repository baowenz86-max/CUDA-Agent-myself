// PyTorch binding for custom ConvTranspose1d, min, erf kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_custom_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void min_scalar_custom_launcher(
    const float* input,
    float* output,
    int size,
    float val,
    cudaStream_t stream);

extern "C" void erf_custom_launcher(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);

// Wrapper for conv_transpose1d
static torch::Tensor conv_transpose1d_custom(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must have shape [N, C_in, L_in]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [C_in, C_out, K]");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape [C_out]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    // Ensure contiguous tensors (needed for raw pointer access)
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1);
    int K = weight.size(2);
    TORCH_CHECK(weight.size(0) == C_in, "weight C_in dimension mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias size mismatch");

    // Output length formula: (L_in - 1) * stride - 2 * padding + K + output_padding
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "computed output length must be positive");

    // Allocate output: [N, C_out, L_out]
    auto output = torch::empty({N, C_out, L_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    conv_transpose1d_custom_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, static_cast<int>(L_out),
        K, static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(output_padding),
        stream);

    return output;
}

// Wrapper for min with scalar
static torch::Tensor min_scalar_custom(torch::Tensor input, double val) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    min_scalar_custom_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        static_cast<float>(val),
        stream);
    return output;
}

// Wrapper for erf
static torch::Tensor erf_custom(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    erf_custom_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream);
    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_custom", &conv_transpose1d_custom,
          "Custom ConvTranspose1d (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("output_padding") = 0);
    m.def("min_scalar_custom", &min_scalar_custom,
          "Custom min with scalar (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("val"));
    m.def("erf_custom", &erf_custom,
          "Custom erf (CUDA)",
          pybind11::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
