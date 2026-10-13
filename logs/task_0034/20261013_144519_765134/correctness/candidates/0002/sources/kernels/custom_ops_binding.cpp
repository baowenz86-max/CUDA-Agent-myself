#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from .cu file
extern "C" void upsample_nearest2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale,
    cudaStream_t stream);

extern "C" void one_hot_launcher(
    const int64_t* input,
    int64_t* output,
    int64_t total_spatial,
    int64_t num_classes,
    cudaStream_t stream);

extern "C" void conv_transpose2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding, int output_padding,
    int H_out, int W_out,
    cudaStream_t stream);

// Helper: get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper: check CUDA float contiguous tensor
static void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ConvTranspose2d implementation (custom CUDA kernel)
torch::Tensor custom_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  // Input checks
  check_cuda_float_contiguous(input, "input");
  check_cuda_float_contiguous(weight, "weight");
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
  }
  TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
  TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in, C_out, kH, kW)");
  TORCH_CHECK(input.size(1) == weight.size(0), "input channels must match weight first dim");

  const int N = static_cast<int>(input.size(0));
  const int C_in = static_cast<int>(input.size(1));
  const int H_in = static_cast<int>(input.size(2));
  const int W_in = static_cast<int>(input.size(3));
  const int C_out = static_cast<int>(weight.size(1));
  const int kH = static_cast<int>(weight.size(2));
  const int kW = static_cast<int>(weight.size(3));

  // Compute output dimensions (same formula as PyTorch)
  int H_out = (H_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kH + static_cast<int>(output_padding);
  int W_out = (W_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kW + static_cast<int>(output_padding);
  TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output size");

  // Allocate output tensor
  auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

  // Get current CUDA stream
  cudaStream_t stream = getCurrentCUDAStream();

  // Get raw pointers
  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
  float* output_ptr = output.data_ptr<float>();

  // Launch kernel
  conv_transpose2d_launcher(
      input_ptr, weight_ptr, bias_ptr, output_ptr,
      N, C_in, H_in, W_in,
      C_out, kH, kW,
      static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(output_padding),
      H_out, W_out,
      stream);

  // Check CUDA errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

  return output;
}

// Upsample Nearest2d
torch::Tensor custom_upsample_nearest2d(torch::Tensor input, double scale_factor) {
  check_cuda_float_contiguous(input, "input");
  TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

  const int N = static_cast<int>(input.size(0));
  const int C = static_cast<int>(input.size(1));
  const int H = static_cast<int>(input.size(2));
  const int W = static_cast<int>(input.size(3));
  const int scale = static_cast<int>(scale_factor);
  TORCH_CHECK(scale > 0, "scale_factor must be positive");

  // Output size
  const int out_H = H * scale;
  const int out_W = W * scale;
  auto output = torch::empty({N, C, out_H, out_W}, input.options());

  cudaStream_t stream = getCurrentCUDAStream();
  upsample_nearest2d_launcher(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      N, C, H, W, scale, stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
  return output;
}

// One-hot
torch::Tensor custom_one_hot(torch::Tensor target, int64_t num_classes) {
  TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
  TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
  TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
  TORCH_CHECK(target.dim() >= 1, "target must have at least 1 dim");
  TORCH_CHECK(num_classes > 0, "num_classes must be positive");

  // Output shape: target.shape + [num_classes]
  std::vector<int64_t> out_shape(target.sizes().begin(), target.sizes().end());
  out_shape.push_back(num_classes);
  auto output = torch::empty(out_shape, target.options());

  int64_t total_spatial = target.numel();
  cudaStream_t stream = getCurrentCUDAStream();
  one_hot_launcher(
      target.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      total_spatial,
      num_classes,
      stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
  return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
  m.def("custom_conv_transpose2d", &custom_conv_transpose2d,
        "Custom ConvTranspose2d using custom CUDA kernel",
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
  m.def("custom_upsample_nearest2d", &custom_upsample_nearest2d,
        "Custom nearest neighbor upsample",
        py::arg("input"), py::arg("scale_factor"));
  m.def("custom_one_hot", &custom_one_hot,
        "Custom one_hot",
        py::arg("target"), py::arg("num_classes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
