// PyTorch binding for custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log10_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void lt_scalar_launcher(const float* input, uint8_t* output, float threshold, int64_t numel, cudaStream_t stream);
extern "C" void bool_to_float_launcher(const uint8_t* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void conv_transpose1d_launcher(const float* input, const float* weight, const float* conv_bias, float* output,
                                          int batch_size, int in_channels, int out_channels, int input_length, int output_length,
                                          int kernel_size, int stride, int padding, cudaStream_t stream);
extern "C" void add_bias_launcher(const float* input, const float* bias, float* output, int batch_size, int channels, int width, cudaStream_t stream);

// Utility to check CUDA tensor
static void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must have dtype torch.float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_cuda_bool(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBool, name, " must have dtype torch.bool");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// log10_forward: y = log10(x)
torch::Tensor log10_forward(torch::Tensor x) {
    check_cuda_float(x, "log10_forward input");
    auto y = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log10_launcher(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), stream);
    return y;
}

// lt_scalar_forward: y = (x < threshold) as bool tensor
torch::Tensor lt_scalar_forward(torch::Tensor x, double threshold) {
    check_cuda_float(x, "lt_scalar_forward input");
    // PyTorch bool tensor uses uint8 storage; allocate with dtype bool.
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    lt_scalar_launcher(x.data_ptr<float>(),
                       reinterpret_cast<uint8_t*>(y.data_ptr<bool>()),
                       static_cast<float>(threshold),
                       x.numel(),
                       stream);
    return y;
}

// boolean_to_float: convert bool tensor to float (0.0/1.0)
torch::Tensor boolean_to_float(torch::Tensor x) {
    check_cuda_bool(x, "boolean_to_float input");
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bool_to_float_launcher(reinterpret_cast<const uint8_t*>(x.data_ptr<bool>()),
                           y.data_ptr<float>(),
                           x.numel(),
                           stream);
    return y;
}

// conv_transpose1d_forward: custom ConvTranspose1d without adding additional bias
// The parameters are: weight (in_channels, out_channels, kernel_size), conv_bias (out_channels)
// stride, padding, output_padding are ints (or tuples with single element).
// We assume 1D and scalar stride/padding/output_padding.
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor conv_bias,
                                       int stride,
                                       int padding,
                                       int output_padding) {
    check_cuda_float(input, "conv_transpose1d_forward input");
    check_cuda_float(weight, "conv_transpose1d_forward weight");
    TORCH_CHECK(conv_bias.dim() == 1 && conv_bias.is_cuda() && conv_bias.dtype() == torch::kFloat32 && conv_bias.is_contiguous(),
                "conv_bias must be CUDA float32 contiguous 1D tensor");

    // Input shape: (batch_size, in_channels, input_length)
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight first dim must match input channels");
    TORCH_CHECK(conv_bias.size(0) == out_channels, "conv_bias size must match out_channels");

    // Compute output length for ConvTranspose1d:
    // output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output_length must be positive");

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(input.data_ptr<float>(),
                              weight.data_ptr<float>(),
                              conv_bias.data_ptr<float>(),
                              output.data_ptr<float>(),
                              batch_size,
                              in_channels,
                              out_channels,
                              input_length,
                              output_length,
                              kernel_size,
                              stride,
                              padding,
                              stream);
    return output;
}

// add_bias_forward: add bias of shape (channels, 1) to input (batch, channels, width)
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float(input, "add_bias_forward input");
    TORCH_CHECK(bias.dim() == 2 && bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.is_contiguous(),
                "bias must be CUDA float32 contiguous 2D tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, width)");
    int batch_size = input.size(0);
    int channels = input.size(1);
    int width = input.size(2);
    TORCH_CHECK(bias.size(0) == channels, "bias first dim must match channels");
    TORCH_CHECK(bias.size(1) == 1, "bias second dim must be 1");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(input.data_ptr<float>(),
                      bias.data_ptr<float>(),
                      output.data_ptr<float>(),
                      batch_size,
                      channels,
                      width,
                      stream);
    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("log10_forward", &log10_forward, "Compute log10 elementwise", pybind11::arg("x"));
    m.def("lt_scalar_forward", &lt_scalar_forward, "Compute x < threshold, returns bool tensor",
          pybind11::arg("x"), pybind11::arg("threshold"));
    m.def("boolean_to_float", &boolean_to_float, "Convert bool tensor to float (0/1)", pybind11::arg("x"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward without extra bias",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("conv_bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias of shape (C,1) broadcast across width",
          pybind11::arg("input"), pybind11::arg("bias"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
