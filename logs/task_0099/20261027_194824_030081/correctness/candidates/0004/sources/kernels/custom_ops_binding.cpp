#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void fused_sqrt_round_launcher(float* output,
                                          const float* input,
                                          long long numel,
                                          cudaStream_t stream);

extern "C" void conv_transpose2d_launcher(const float* input,
                                          const float* weight,
                                          const float* bias,
                                          float* output,
                                          int N, int C_in, int H_in, int W_in,
                                          int C_out, int kH, int kW,
                                          int stride, int padding, int output_padding,
                                          int H_out, int W_out,
                                          cudaStream_t stream);

// Wrapper for fused_sqrt_round
torch::Tensor fused_sqrt_round(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
  auto x_contig = x.contiguous();
  auto output = torch::empty_like(x_contig);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  fused_sqrt_round_launcher(output.data_ptr<float>(),
                            x_contig.data_ptr<float>(),
                            x_contig.numel(),
                            stream);
  if (!x.is_contiguous()) {
    output = output.as_strided(x.sizes(), x.strides());
  }
  return output;
}

// Wrapper for conv_transpose2d_forward using custom kernel
torch::Tensor conv_transpose2d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      c10::optional<torch::Tensor> bias,
                                      int64_t stride,
                                      int64_t padding,
                                      int64_t output_padding) {
  // Validate inputs
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
  }
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
  TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in,C_out,kH,kW)");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
  }

  int64_t N = input.size(0);
  int64_t C_in = input.size(1);
  int64_t H_in = input.size(2);
  int64_t W_in = input.size(3);
  int64_t C_in_weight = weight.size(0);
  int64_t C_out = weight.size(1);
  int64_t kH = weight.size(2);
  int64_t kW = weight.size(3);
  TORCH_CHECK(C_in == C_in_weight, "input channels must match weight");
  if (bias.has_value()) {
    TORCH_CHECK(C_out == bias.value().size(0), "bias size must match output channels");
  }

  int64_t H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
  int64_t W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;
  TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output size");

  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  auto bias_contig = bias.has_value() ? bias.value().contiguous() : torch::Tensor();

  auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard guard(input.device());

  const float* bias_ptr = bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr;

  conv_transpose2d_launcher(
      input_contig.data_ptr<float>(),
      weight_contig.data_ptr<float>(),
      bias_ptr,
      output.data_ptr<float>(),
      static_cast<int>(N),
      static_cast<int>(C_in),
      static_cast<int>(H_in),
      static_cast<int>(W_in),
      static_cast<int>(C_out),
      static_cast<int>(kH),
      static_cast<int>(kW),
      static_cast<int>(stride),
      static_cast<int>(padding),
      static_cast<int>(output_padding),
      static_cast<int>(H_out),
      static_cast<int>(W_out),
      stream);

  return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
  m.def("fused_sqrt_round", &fused_sqrt_round,
        "Fused sqrt then round",
        pybind11::arg("x"));
  m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
        "ConvTranspose2d forward via custom kernel",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
