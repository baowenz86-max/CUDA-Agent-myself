#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
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
    cudaStream_t stream);

extern "C" void median_launcher(
    const float* input,
    float* output_median,
    int num_elements,
    cudaStream_t stream);

extern "C" void ne_float_scalar_launcher(
    const float* input,
    float scalar,
    unsigned char* output,
    int num_elements,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper for conv_transpose1d_forward
// Args: input [batch, in_channels, in_length] float32 CUDA
//       weight [in_channels, out_channels, kernel_size] float32 CUDA
//       bias [out_channels] float32 CUDA
//       stride, padding, output_padding ints
// Returns: output [batch, out_channels, out_length] float32 CUDA
static torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Check CUDA
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "conv_transpose1d_forward: bias must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: input dtype must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: weight dtype must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: bias dtype must be float32");

    // Check contiguity
    TORCH_CHECK(input.is_contiguous(), "conv_transpose1d_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "conv_transpose1d_forward: bias must be contiguous");

    // Check dimensions
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "conv_transpose1d_forward: bias must be 1D (out_channels)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "conv_transpose1d_forward: weight in_channels dimension mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "conv_transpose1d_forward: bias size mismatch");

    // Compute output length: out_length = (in_length - 1) * stride - 2*padding + kernel_size + output_padding
    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "conv_transpose1d_forward: computed output length must be positive");

    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());

    cudaStream_t stream = getCurrentStream();

    conv_transpose1d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        static_cast<int>(out_length),
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    return output;
}

// Wrapper for median_forward
// Args: input tensor (any shape) float32 CUDA
// Returns: scalar tensor float32 CUDA
static torch::Tensor median_forward(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "median_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "median_forward: input dtype must be float32");
    TORCH_CHECK(input.is_contiguous(), "median_forward: input must be contiguous");
    TORCH_CHECK(input.numel() > 0, "median_forward: input must have at least one element");

    auto output = torch::empty({}, input.options());

    cudaStream_t stream = getCurrentStream();
    median_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream
    );

    return output;
}

// Wrapper for ne_forward
// Args: input tensor float32 CUDA, other can be float scalar or 0-dim tensor
// Returns: bool tensor same shape as input
static torch::Tensor ne_forward(torch::Tensor input, torch::Scalar other) {
    TORCH_CHECK(input.is_cuda(), "ne_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "ne_forward: input dtype must be float32");
    TORCH_CHECK(input.is_contiguous(), "ne_forward: input must be contiguous");

    float scalar_val;
    if (other.isFloatingPoint()) {
        scalar_val = static_cast<float>(other.toDouble());
    } else {
        TORCH_CHECK(false, "ne_forward: other must be a float scalar");
    }

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));

    cudaStream_t stream = getCurrentStream();
    ne_float_scalar_launcher(
        input.data_ptr<float>(),
        scalar_val,
        reinterpret_cast<unsigned char*>(output.data_ptr<bool>()),
        static_cast<int>(input.numel()),
        stream
    );

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("median_forward", &median_forward,
          "Median forward",
          py::arg("input"));
    m.def("ne_forward", &ne_forward,
          "Not equal forward",
          py::arg("input"), py::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
