#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int length,
                                        int L_out,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding,
                                        cudaStream_t stream);

extern "C" void layer_norm_launcher(const float* x,
                                    const float* weight,
                                    const float* bias,
                                    float* y,
                                    int batch,
                                    int out_channels,
                                    int length,
                                    float eps,
                                    cudaStream_t stream);

extern "C" void addcmul_launcher(const float* input,
                                 const float* tensor1,
                                 const float* tensor2,
                                 float* output,
                                 int total_elements,
                                 float value,
                                 cudaStream_t stream);

// -------------------- ConvTranspose1d wrapper --------------------

torch::Tensor conv_transpose_forward_cuda(torch::Tensor x,
                                         torch::Tensor weight,
                                         torch::Tensor bias,
                                         int64_t stride,
                                         int64_t padding,
                                         int64_t output_padding) {
  TORCH_CHECK(x.is_cuda(), "conv_transpose_forward: x must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "conv_transpose_forward: weight must be CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "conv_transpose_forward: x must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose_forward: weight must be float32");
  TORCH_CHECK(x.dim() == 3, "conv_transpose_forward: x must be 3D [batch, in_channels, length]");
  TORCH_CHECK(weight.dim() == 3, "conv_transpose_forward: weight must be 3D [in_channels, out_channels, kernel_size]");

  // Make contiguous
  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();

  int64_t batch = x_contig.size(0);
  int64_t in_channels = x_contig.size(1);
  int64_t length = x_contig.size(2);
  int64_t out_channels = weight_contig.size(1);
  int64_t kernel_size = weight_contig.size(2);

  // Compute output length as PyTorch ConvTranspose1d: L_out = (length - 1)*stride - 2*padding + kernel_size + output_padding
  int64_t L_out = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
  TORCH_CHECK(L_out > 0, "conv_transpose_forward: computed output length must be positive");

  // Optional bias handling
  torch::Tensor bias_contig;
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "conv_transpose_forward: bias must be CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose_forward: bias must be float32");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels,
                "conv_transpose_forward: bias shape must be [out_channels]");
    bias_contig = bias.contiguous();
  }

  // Allocate output tensor [batch, out_channels, L_out], initialized to zero, because the naive kernel may not write elements for invalid transposed positions, but in this case with stride=2, padding=1, all output positions are valid? Actually output_padding>0 adds extra positions at the end that the naive convolution will not compute because the input index would be out of range; those positions should be zero (before bias addition). To ensure correct output, we zero the entire output first.
  auto output = torch::zeros({batch, out_channels, L_out}, x_contig.options());

  // Get current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // If bias exists, we will zero only the extra output_padding positions? No, we zero the whole output, then the kernel will overwrite valid positions and add bias to all positions (including padding) if bias exists. But PyTorch adds bias to all positions, including the output_padding extra zeros, so it's correct. However, our kernel adds bias unconditionally to every output element when bias is not null. That matches PyTorch behavior: bias is added to all L_out positions. But we need to ensure that the extra output_padding positions (where no valid transposed conv sum exists) are zero before bias addition. Since we zero the entire output and the kernel only writes when bias is null or when there is a valid sum? But the kernel always writes sum (which is 0 + bias if bias present) for all ol in [0, L_out). The kernel will compute sum=0 for invalid positions (because the inner loops do not add anything) and then add bias, so those positions become bias value. But PyTorch adds bias to those positions as well, so it's correct: the extra output_padding positions are zero from conv then bias added, giving bias value. Our zero-initialized output is overwritten by the kernel for all ol, so the initial zeroing is not strictly needed, but harmless. However, if bias is null, the kernel writes sum=0 for invalid positions, which is correct (zero padding). The kernel writes every output position, so no need for zero initialization. Actually, the kernel will write output[output_idx] = sum for every ol within bounds, so all positions are written. Sum for invalid positions is zero (or bias). So we can use torch::empty and avoid initialization. But to be safe, we keep torch::zeros; it adds a small overhead.

  // Call launcher
  conv_transpose_launcher(
      x_contig.data_ptr<float>(),
      weight_contig.data_ptr<float>(),
      bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(in_channels),
      static_cast<int>(out_channels),
      static_cast<int>(length),
      static_cast<int>(L_out),
      static_cast<int>(kernel_size),
      static_cast<int>(stride),
      static_cast<int>(padding),
      static_cast<int>(output_padding),
      stream
  );

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "conv_transpose_forward: kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

// -------------------- LayerNorm wrapper --------------------

torch::Tensor layer_norm_forward_cuda(torch::Tensor x,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      double eps) {
  TORCH_CHECK(x.is_cuda(), "layer_norm_forward: x must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "layer_norm_forward: weight must be CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "layer_norm_forward: bias must be CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "layer_norm_forward: x must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "layer_norm_forward: weight must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "layer_norm_forward: bias must be float32");
  TORCH_CHECK(x.dim() == 3, "layer_norm_forward: x must be 3D [batch, out_channels, length]");

  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  auto bias_contig = bias.contiguous();

  int64_t batch = x_contig.size(0);
  int64_t out_channels = x_contig.size(1);
  int64_t length = x_contig.size(2);

  TORCH_CHECK(weight_contig.numel() == out_channels * length, "layer_norm_forward: weight shape mismatch");
  TORCH_CHECK(bias_contig.numel() == out_channels * length, "layer_norm_forward: bias shape mismatch");

  auto output = torch::empty_like(x_contig);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  layer_norm_launcher(
      x_contig.data_ptr<float>(),
      weight_contig.data_ptr<float>(),
      bias_contig.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(out_channels),
      static_cast<int>(length),
      static_cast<float>(eps),
      stream
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "layer_norm_forward: kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

// -------------------- Addcmul wrapper --------------------

torch::Tensor addcmul_forward_cuda(torch::Tensor input,
                                   torch::Tensor tensor1,
                                   torch::Tensor tensor2,
                                   double value) {
  TORCH_CHECK(input.is_cuda(), "addcmul_forward: input must be CUDA tensor");
  TORCH_CHECK(tensor1.is_cuda(), "addcmul_forward: tensor1 must be CUDA tensor");
  TORCH_CHECK(tensor2.is_cuda(), "addcmul_forward: tensor2 must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "addcmul_forward: input must be float32");
  TORCH_CHECK(tensor1.dtype() == torch::kFloat32, "addcmul_forward: tensor1 must be float32");
  TORCH_CHECK(tensor2.dtype() == torch::kFloat32, "addcmul_forward: tensor2 must be float32");
  TORCH_CHECK(input.sizes() == tensor1.sizes() && tensor1.sizes() == tensor2.sizes(),
              "addcmul_forward: all tensors must have the same shape");

  auto input_contig = input.contiguous();
  auto tensor1_contig = tensor1.contiguous();
  auto tensor2_contig = tensor2.contiguous();

  auto output = torch::empty_like(input_contig);
  int64_t total_elements = input_contig.numel();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  addcmul_launcher(
      input_contig.data_ptr<float>(),
      tensor1_contig.data_ptr<float>(),
      tensor2_contig.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(total_elements),
      static_cast<float>(value),
      stream
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "addcmul_forward: kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

// -------------------- PyBind Registration --------------------

void register_custom_ops(pybind11::module& m) {
  m.def("conv_transpose_forward", &conv_transpose_forward_cuda,
        "ConvTranspose1d forward",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
  m.def("layer_norm_forward", &layer_norm_forward_cuda,
        "LayerNorm forward",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps"));
  m.def("addcmul_forward", &addcmul_forward_cuda,
        "Addcmul forward",
        pybind11::arg("input"), pybind11::arg("tensor1"), pybind11::arg("tensor2"), pybind11::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
