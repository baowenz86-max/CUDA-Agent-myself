// kernels/custom_ops_binding.cpp
// PyTorch bindings for custom_conv_transpose1d and custom_histc.
// This file handles tensor checks, allocation, and launcher invocation.
// It does not contain any CUDA kernel definitions or launch syntax.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C_in,
    int C_out,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    cudaStream_t stream
);

extern "C" void custom_histc_launcher(
    const float* input,
    float* output,
    int total_elements,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// custom_conv_transpose1d
// -----------------------------------------------------------------------------
torch::Tensor custom_conv_transpose1d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation
) {
    // Check input tensors
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    }

    // Input dimensions: [N, C_in, L_in]
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L_in)");
    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);

    // Weight: [C_in, C_out/groups, kernel_size]
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    int C_out = weight.size(1) * groups;
    int kernel_size = weight.size(2);

    // Verify weight first dimension matches C_in
    TORCH_CHECK(weight.size(0) == C_in, "weight size mismatch");

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    TORCH_CHECK(L_out > 0, "computed output length must be positive");

    // Allocate output tensor
    auto output = torch::empty({N, C_out, L_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N,
        C_in,
        C_out,
        L_in,
        L_out,
        kernel_size,
        (int)stride,
        (int)padding,
        (int)output_padding,
        (int)groups,
        (int)dilation,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// custom_histc
// -----------------------------------------------------------------------------
torch::Tensor custom_histc(
    torch::Tensor input,
    int64_t bins,
    double min,
    double max
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bins > 0, "bins must be positive");

    int64_t total_elements = input.numel();

    // Allocate output tensor
    auto output = torch::empty({bins}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_histc_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)total_elements,
        (int)bins,
        (float)min,
        (float)max,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
          "Custom ConvTranspose1d using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"),
          pybind11::arg("groups"),
          pybind11::arg("dilation"));

    m.def("custom_histc", &custom_histc,
          "Custom histogram",
          pybind11::arg("input"),
          pybind11::arg("bins"),
          pybind11::arg("min"),
          pybind11::arg("max"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
