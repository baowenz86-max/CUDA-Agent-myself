#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out_base,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void custom_minimum_launcher(
    const float* x,
    const float* y,
    float* out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long y_channel_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride,
    cudaStream_t stream);

extern "C" void custom_floor_launcher(
    const float* x,
    float* out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride,
    cudaStream_t stream);

// custom_conv_transpose1d binding
torch::Tensor custom_conv_transpose1d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  }
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  if (bias.defined()) {
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  }
  TORCH_CHECK(x.dim() == 3, "x must be 3D [N, C_in, L]");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_in, C_out, K]");
  TORCH_CHECK(x.size(1) == weight.size(0), "input channels mismatch");
  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(1), "bias size mismatch");
  }
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous for cuDNN");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous for cuDNN");
  if (bias.defined()) {
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous for cuDNN");
  }

  int64_t batch = x.size(0);
  int64_t in_channels = x.size(1);
  int64_t length_in = x.size(2);
  int64_t out_channels = weight.size(1);
  int64_t kernel_size = weight.size(2);

  int64_t length_out_base = (length_in - 1) * stride - 2 * padding + kernel_size;
  TORCH_CHECK(length_out_base > 0, "base output length must be positive");
  int64_t length_out = length_out_base + output_padding;
  TORCH_CHECK(length_out > 0, "output length must be positive");

  // Allocate output tensor; shape [batch, out_channels, length_out]
  auto output = torch::empty({batch, out_channels, length_out}, x.options());
  if (output_padding > 0) {
    // Zero initialize so padding region is zero
    output.zero_();
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Call launcher
  custom_conv_transpose1d_launcher(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(in_channels),
      static_cast<int>(out_channels),
      static_cast<int>(length_in),
      static_cast<int>(length_out_base),
      static_cast<int>(length_out),
      static_cast<int>(kernel_size),
      static_cast<int>(stride),
      static_cast<int>(padding),
      static_cast<int>(output_padding),
      stream);

  return output;
}

// custom_minimum binding
torch::Tensor custom_minimum(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(y.is_cuda(), "y must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.dim() == 3, "x must be 3D");
  TORCH_CHECK(y.dim() == 2, "y must be 2D");
  TORCH_CHECK(y.size(0) == x.size(1), "y size[0] must match x channels");
  TORCH_CHECK(y.size(1) == 1, "y size[1] must be 1");

  auto out = torch::empty_like(x);

  long long batch = x.size(0);
  long long channels = x.size(1);
  long long length = x.size(2);

  long long x_batch_stride = x.stride(0);
  long long x_channel_stride = x.stride(1);
  long long x_length_stride = x.stride(2);
  long long y_channel_stride = y.stride(0);
  long long out_batch_stride = out.stride(0);
  long long out_channel_stride = out.stride(1);
  long long out_length_stride = out.stride(2);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  custom_minimum_launcher(
      x.data_ptr<float>(),
      y.data_ptr<float>(),
      out.data_ptr<float>(),
      batch, channels, length,
      x_batch_stride, x_channel_stride, x_length_stride,
      y_channel_stride,
      out_batch_stride, out_channel_stride, out_length_stride,
      stream);

  return out;
}

// custom_floor binding
torch::Tensor custom_floor(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 3, "x must be 3D");

  auto out = torch::empty_like(x);

  long long batch = x.size(0);
  long long channels = x.size(1);
  long long length = x.size(2);

  long long x_batch_stride = x.stride(0);
  long long x_channel_stride = x.stride(1);
  long long x_length_stride = x.stride(2);
  long long out_batch_stride = out.stride(0);
  long long out_channel_stride = out.stride(1);
  long long out_length_stride = out.stride(2);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  custom_floor_launcher(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      batch, channels, length,
      x_batch_stride, x_channel_stride, x_length_stride,
      out_batch_stride, out_channel_stride, out_length_stride,
      stream);

  return out;
}

void register_custom_ops(pybind11::module& m) {
  m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
        "Custom ConvTranspose1d using cuDNN",
        py::arg("x"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
  m.def("custom_minimum", &custom_minimum,
        "Custom minimum with broadcasting",
        py::arg("x"), py::arg("y"));
  m.def("custom_floor", &custom_floor,
        "Custom floor",
        py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
