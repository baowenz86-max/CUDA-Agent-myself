#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias,
                                  int64_t N, int64_t C, int64_t L, cudaStream_t stream);
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int64_t N, int64_t C_in, int64_t L_in, int64_t C_out, int64_t K,
    int64_t stride, int64_t padding, int64_t output_padding, cudaStream_t stream);
extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t C_in, int64_t L_in, int64_t C_out, int64_t K,
    int64_t stride, int64_t padding, cudaStream_t stream);

namespace {

cudaStream_t getCurrentStream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

} // namespace

// ConvTranspose1d forward wrapper
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
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels/groups, kernel_size)");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(1);
  const int64_t K = weight.size(2);
  TORCH_CHECK(weight.size(0) == C_in, "weight first dimension must match input channels");

  // Compute output length using formula (L_in - 1) * stride - 2 * padding + K + output_padding
  int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());
  cudaStream_t stream = getCurrentStream();

  conv_transpose1d_launcher(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      N, C_in, L_in, C_out, K,
      stride, padding, output_padding,
      stream);

  return output;
}

// Add bias broadcast wrapper
torch::Tensor add_bias_forward(const torch::Tensor& input, const torch::Tensor& bias) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(bias.dim() == 2, "bias must be 2D (C, 1)");

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t L = input.size(2);
  TORCH_CHECK(bias.size(0) == C, "bias first dimension must match input channels");
  TORCH_CHECK(bias.size(1) == 1, "bias second dimension must be 1");

  auto output = torch::empty_like(input);
  cudaStream_t stream = getCurrentStream();

  add_bias_launcher(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      bias.data_ptr<float>(),
      N, C, L,
      stream);

  return output;
}

// Conv1d forward wrapper
torch::Tensor conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding) {

  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
  }
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels/groups, kernel_size)");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(0);
  const int64_t K = weight.size(2);
  TORCH_CHECK(weight.size(1) == C_in, "weight second dimension must match input channels");

  // Compute output length: (L_in - K) / stride + 1 (since padding=0)
  int64_t L_out = (L_in - K) / stride + 1;
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());
  cudaStream_t stream = getCurrentStream();

  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  conv1d_launcher(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      output.data_ptr<float>(),
      N, C_in, L_in, C_out, K,
      stride, padding,
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
