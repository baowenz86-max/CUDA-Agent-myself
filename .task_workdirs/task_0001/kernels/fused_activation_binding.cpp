#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from fused_activation.cu
extern "C" void fused_activation_float_launcher(
    float* output,
    const float* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream);

extern "C" void fused_activation_double_launcher(
    double* output,
    const double* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream);

extern "C" void fused_activation_half_launcher(
    __half* output,
    const __half* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream);

// PyTorch wrapper for fused activation
torch::Tensor fused_activation_forward(
    torch::Tensor input,
    double leakyrelu_negative_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    // Allocate output tensor with same shape and dtype
    auto output = torch::empty_like(input);
    
    int numel = static_cast<int>(input.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Dispatch based on dtype
    if (input.dtype() == torch::kFloat32) {
        fused_activation_float_launcher(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            numel,
            leakyrelu_negative_slope,
            pow_exponent,
            mish_alpha,
            relu_threshold,
            abs_max,
            stream);
    } else if (input.dtype() == torch::kFloat64) {
        fused_activation_double_launcher(
            output.data_ptr<double>(),
            input.data_ptr<double>(),
            numel,
            leakyrelu_negative_slope,
            pow_exponent,
            mish_alpha,
            relu_threshold,
            abs_max,
            stream);
    } else if (input.dtype() == torch::kFloat16) {
        fused_activation_half_launcher(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            numel,
            leakyrelu_negative_slope,
            pow_exponent,
            mish_alpha,
            relu_threshold,
            abs_max,
            stream);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_activation: only float32, float64, and float16 are supported");
    }
    
    return output;
}

// Registration function
void register_fused_activation(pybind11::module& m) {
    m.def("fused_activation_forward", &fused_activation_forward,
          "Fused activation: LeakyReLU -> Power -> Abs -> Mish -> ReLU -> Clamp",
          py::arg("input"),
          py::arg("leakyrelu_negative_slope"),
          py::arg("pow_exponent"),
          py::arg("mish_alpha"),
          py::arg("relu_threshold"),
          py::arg("abs_max"));
}

// Auto-register via macro
REGISTER_BINDING(fused_activation, register_fused_activation);
