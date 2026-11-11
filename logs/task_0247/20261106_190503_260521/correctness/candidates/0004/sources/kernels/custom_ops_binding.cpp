#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Forward declarations of extern "C" launchers from custom_ops.cu
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding,
    cudaStream_t stream);

// Helper to get current CUDA stream
cudaStream_t getCurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

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
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
  TORCH_CHECK(stride >= 1, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");
  TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(1); // ConvTranspose weight: [in_channels, out_channels, kernel_size]
  const int64_t K = weight.size(2);

  TORCH_CHECK(weight.size(0) == C_in, "weight first dimension must match input channels");
  TORCH_CHECK(K > 0, "kernel size must be positive");

  int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());

  cudaStream_t stream = getCurrentStream();

  conv_transpose1d_launcher(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(N), static_cast<int>(C_in), static_cast<int>(L_in),
      static_cast<int>(C_out), static_cast<int>(K),
      static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(output_padding),
      stream);

  return output;
}

// Add bias forward: bias shape [out_channels, 1] broadcast to [batch, out_channels, length]
 torch::Tensor add_bias_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(bias.dim() == 2, "bias must be 2D (C, 1)");
  const int64_t C = input.size(1);
  const int64_t L = input.size(2);
  TORCH_CHECK(bias.size(0) == C, "bias first dimension must match input channels");
  TORCH_CHECK(bias.size(1) == 1, "bias second dimension must be 1");

  auto output = torch::empty_like(input);

  cudaStream_t stream = getCurrentStream();

  // We need a launcher for add_bias; we'll declare and use a kernel from custom_ops.cu
  // For simplicity, we can implement add_bias as a custom kernel in custom_ops.cu and declare it here.
  // However, the .cu already includes a bias-add kernel (add_bias_kernel) but not a launcher for generic add_bias_forward.
  // We can add a launcher in .cu: extern "C" void add_bias_launcher(...) and call it here.
  // For now, we can use the same kernel launched via a simple launcher in .cu.
  // But we must not add new .cu code in this phase? We can modify .cu as needed.
  // Since we are only providing .cpp, we need to ensure .cu has a corresponding launcher.
  // Let's define add_bias_launcher in .cu and declare it here.
  extern "C" void add_bias_launcher(
      float* output, const float* input, const float* bias,
      int N, int C, int L,
      cudaStream_t stream);

  add_bias_launcher(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      bias.data_ptr<float>(),
      static_cast<int>(input.size(0)), static_cast<int>(C), static_cast<int>(L),
      stream);

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
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
  TORCH_CHECK(stride >= 1, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");

  const int64_t N = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t L_in = input.size(2);
  const int64_t C_out = weight.size(0);
  const int64_t K = weight.size(2);

  TORCH_CHECK(weight.size(1) == C_in, "weight second dimension must match input channels");
  TORCH_CHECK(K > 0, "kernel size must be positive");

  int64_t L_out = (L_in + 2 * padding - K) / stride + 1;
  TORCH_CHECK(L_out > 0, "output length must be positive");

  auto output = torch::empty({N, C_out, L_out}, input.options());

  cudaStream_t stream = getCurrentStream();

  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

  conv1d_launcher(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      output.data_ptr<float>(),
      static_cast<int>(N), static_cast<int>(C_in), static_cast<int>(L_in),
      static_cast<int>(C_out), static_cast<int>(K),
      static_cast<int>(stride), static_cast<int>(padding),
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
