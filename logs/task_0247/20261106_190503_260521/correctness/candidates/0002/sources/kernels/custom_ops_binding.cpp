// Binding implementations for conv_transpose1d_forward, add_bias_forward, conv1d_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, bool has_bias,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    float* output, const float* input, const float* bias,
    int64_t N, int64_t C, int64_t L,
    cudaStream_t stream);

namespace {

cudaStream_t getCurrentStream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

int64_t getConvTransposeOutputLength(int64_t L_in, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding) {
  return (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
}

int64_t getConv1dOutputLength(int64_t L_in, int64_t kernel_size, int64_t stride, int64_t padding) {
  return (L_in - kernel_size) / stride + 1;
}

} // namespace

// ConvTranspose1d forward
torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
  TORCH_CHECK(stride >= 1, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");
  TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(1);
  const int64_t kernel_size = weight.size(2);

  TORCH_CHECK(weight.size(0) == C_in, "weight first dim must match C_in");

  int64_t L_out = getConvTransposeOutputLength(L_in, kernel_size, stride, padding, output_padding);
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());

  cudaStream_t stream = getCurrentStream();
  conv_transpose1d_launcher(
      input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
      (int)N, (int)C_in, (int)C_out, (int)L_in, (int)L_out,
      (int)kernel_size, (int)stride, (int)padding, (int)output_padding,
      stream);

  return output;
}

// Add bias forward
torch::Tensor add_bias_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D");
  TORCH_CHECK(bias.dim() == 2, "bias must be 2D");
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t L = input.size(2);
  TORCH_CHECK(bias.size(0) == C, "bias first dim must match C");
  TORCH_CHECK(bias.size(1) == 1, "bias second dim must be 1");

  auto output = torch::empty_like(input);

  cudaStream_t stream = getCurrentStream();
  add_bias_launcher(
      output.data_ptr<float>(), input.data_ptr<float>(), bias.data_ptr<float>(),
      N, C, L, stream);

  return output;
}

// Conv1d forward
torch::Tensor conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
  }
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
  TORCH_CHECK(stride >= 1, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(0);
  const int64_t kernel_size = weight.size(2);

  TORCH_CHECK(weight.size(1) == C_in, "weight second dim must match C_in");

  int64_t L_out = getConv1dOutputLength(L_in, kernel_size, stride, padding);
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());

  cudaStream_t stream = getCurrentStream();
  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  conv1d_launcher(
      input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
      (int)N, (int)C_in, (int)C_out, (int)L_in, (int)L_out,
      (int)kernel_size, (int)stride, (int)padding, bias.has_value(),
      stream);

  return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
  m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
        "ConvTranspose1d forward using cuDNN",
        py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
  m.def("add_bias_forward", &add_bias_forward,
        "Add bias broadcast forward",
        py::arg("input"), py::arg("bias"));
  m.def("conv1d_forward", &conv1d_forward,
        "Conv1d forward using cuDNN",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
