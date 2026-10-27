// Custom ops binding for ConvTranspose1d and Flatten
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "/usr/local/cuda/include/cudnn.h"
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Helper to check CUDA errors
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// Helper to check cuDNN errors
#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
    } \
  } while (0)

// ConvTranspose1d forward using cuDNN
// Input tensor shape: (N, C_in, L_in)
// Weight tensor shape: (C_in, C_out, kernel_size)
// Bias tensor shape: (C_out) optional
// stride, padding, output_padding are ints
// Returns tensor shape: (N, C_out, L_out)
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  // Validate inputs
  TORCH_CHECK(input.is_cuda(), "conv_transpose1d input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "conv_transpose1d weight must be CUDA tensor");
  TORCH_CHECK(!bias.defined() || bias.is_cuda(), "conv_transpose1d bias must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose1d input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d weight must be float32");
  TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "conv_transpose1d bias must be float32");
  TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions (N, C, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must have 3 dimensions (C_in, C_out, kernel)");

  // Ensure contiguous tensors (cuDNN requires contiguous)
  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

  int64_t N = input_contig.size(0);
  int64_t C_in = input_contig.size(1);
  int64_t L_in = input_contig.size(2);
  int64_t C_out = weight_contig.size(1);
  int64_t kernel_size = weight_contig.size(2);

  TORCH_CHECK(weight_contig.size(0) == C_in, "weight shape mismatch: expected in_channels ", C_in, " but got ", weight_contig.size(0));
  if (bias_contig.defined()) {
    TORCH_CHECK(bias_contig.size(0) == C_out, "bias size must match out_channels ", C_out);
  }

  // Compute output length
  int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
  TORCH_CHECK(L_out > 0, "calculated output length is non-positive");

  // Allocate output
  auto output = torch::empty({N, C_out, L_out}, input_contig.options());

  // Get current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Create cuDNN handle
  cudnnHandle_t cudnn_handle;
  CUDNN_CHECK(cudnnCreate(&cudnn_handle));
  CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));

  // Note: 1D transposed convolution can be implemented using 2D convolution with height=1
  // We will create tensor descriptors for NCHW with H=1, W=L
  const int dims = 4; // N, C, H, W
  std::vector<int> input_dim = {static_cast<int>(N), static_cast<int>(C_in), 1, static_cast<int>(L_in)};
  std::vector<int> input_stride = {static_cast<int>(C_in * L_in), static_cast<int>(L_in), static_cast<int>(L_in), 1};
  std::vector<int> weight_dim = {static_cast<int>(C_in), static_cast<int>(C_out), 1, static_cast<int>(kernel_size)};
  std::vector<int> output_dim = {static_cast<int>(N), static_cast<int>(C_out), 1, static_cast<int>(L_out)};
  std::vector<int> output_stride = {static_cast<int>(C_out * L_out), static_cast<int>(L_out), static_cast<int>(L_out), 1};

  cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
  cudnnFilterDescriptor_t weight_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, dims, input_dim.data(), input_stride.data()));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, dims, output_dim.data(), output_stride.data()));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dims, weight_dim.data()));
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc, 2, // array length for pad/stride/dilation
                                             std::vector<int>{0, static_cast<int>(padding)}.data(),
                                             std::vector<int>{1, static_cast<int>(stride)}.data(),
                                             std::vector<int>{1, 1}.data(),
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Find best backward data algorithm
  cudnnConvolutionBwdDataAlgo_t algo;
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle,
      weight_desc,
      input_desc, // diff desc: dy
      conv_desc,
      output_desc, // grad desc: dx
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      0,
      &algo));

  // Allocate workspace
  size_t workspace_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle,
      weight_desc,
      input_desc,
      conv_desc,
      output_desc,
      algo,
      &workspace_size));
  void* workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  // Perform convolution: output = conv_transpose(input, weight)
  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionBackwardData(
      cudnn_handle,
      &alpha,
      weight_desc,
      weight_contig.data_ptr<float>(),
      input_desc,
      input_contig.data_ptr<float>(),
      conv_desc,
      algo,
      workspace,
      workspace_size,
      &beta,
      output_desc,
      output.data_ptr<float>()));

  // Add bias if provided
  if (bias_contig.defined()) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          1, static_cast<int>(C_out), 1, 1)); // bias shape (1, C_out, 1, 1)
    float alpha_bias = 1.0f;
    float beta_bias = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(cudnn_handle, &alpha_bias, bias_desc, bias_contig.data_ptr<float>(),
                               &beta_bias, output_desc, output.data_ptr<float>()));
  }

  // Cleanup
  if (workspace) cudaFree(workspace);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(weight_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn_handle);

  return output;
}

// Flatten: view as (batch, -1)
torch::Tensor flatten_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "flatten input must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "flatten input must be float32");
  TORCH_CHECK(input.dim() >= 2, "flatten input must have at least 2 dimensions");
  int64_t batch = input.size(0);
  int64_t flat_size = input.numel() / batch;
  return input.reshape({batch, flat_size});
}

void register_custom_ops(pybind11::module& m) {
  m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
        "CUDA ConvTranspose1d forward (cuDNN)",
        pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
  m.def("flatten_forward", &flatten_forward, "Flatten to (batch, -1)",
        pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);