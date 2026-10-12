#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
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
    cudaStream_t stream
);

extern "C" void median_launcher(
    const float* input,
    float* output,
    int total_elements,
    cudaStream_t stream
);

extern "C" void ne_float_scalar_launcher(
    const float* input,
    float scalar,
    bool* output,
    int total_elements,
    cudaStream_t stream
);

// Helper: get current CUDA stream
static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper for conv_transpose1d forward
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");

    // Make contiguous
    input = input.contiguous();
    weight = weight.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Compute output length
    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "invalid output length");

    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());

    // Handle optional bias: if not provided, pass nullptr to launcher
    // but launcher expects non-null? In custom_ops.cu kernel checks if bias != nullptr.
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias->defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == out_channels, "bias shape mismatch");
        bias_contig = b.contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    cudaStream_t stream = get_stream();
    conv_transpose1d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    return output;
}

// Wrapper for median
torch::Tensor median_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");
    input = input.contiguous();

    auto output = torch::empty({}, input.options());  // scalar tensor

    cudaStream_t stream = get_stream();
    median_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream
    );
    return output;
}

// Wrapper for ne (not equal) with float scalar
torch::Tensor ne_forward_float(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    input = input.contiguous();

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    cudaStream_t stream = get_stream();
    ne_float_scalar_launcher(
        input.data_ptr<float>(),
        static_cast<float>(scalar),
        output.data_ptr<bool>(),
        static_cast<int>(input.numel()),
        stream
    );
    return output;
}

// Wrapper for ne (not equal) with tensor scalar (0-dim)
torch::Tensor ne_forward_tensor(torch::Tensor input, torch::Tensor scalar_tensor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(scalar_tensor.is_cuda(), "scalar_tensor must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(scalar_tensor.dtype() == torch::kFloat32, "scalar_tensor must be float32");
    TORCH_CHECK(scalar_tensor.numel() == 1, "scalar_tensor must have exactly one element");
    input = input.contiguous();
    scalar_tensor = scalar_tensor.contiguous();

    // Extract scalar value on CPU (small, can sync)
    float scalar = scalar_tensor.item<float>();
    return ne_forward_float(input, scalar);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("median_forward", &median_forward, "Median forward", py::arg("input"));
    // Register two overloads for ne_forward; Python will dispatch based on second argument type.
    m.def("ne_forward", &ne_forward_float, "NE forward (float scalar)", py::arg("input"), py::arg("scalar"));
    m.def("ne_forward", &ne_forward_tensor, "NE forward (tensor scalar)", py::arg("input"), py::arg("scalar_tensor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
