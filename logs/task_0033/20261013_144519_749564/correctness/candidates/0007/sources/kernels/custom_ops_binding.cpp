#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Launcher from custom_ops.cu
extern "C" void addcmul_forward(
    const float* input,
    const float* tensor1,
    const float* tensor2,
    float* output,
    float value,
    int64_t numel,
    cudaStream_t stream);

#define CHECK_CUDNN(expr)                                                      \
  do {                                                                        \
    cudnnStatus_t status = (expr);                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string("cuDNN error: ") +                 \
                               cudnnGetErrorString(status));                  \
    }                                                                         \
  } while (0)

static cudnnHandle_t get_cudnn_handle() {
  static cudnnHandle_t handle = nullptr;
  if (!handle) {
    CHECK_CUDNN(cudnnCreate(&handle));
  }
  return handle;
}

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

  int64_t batch_size = input.size(0);
  int64_t in_channels = input.size(1);
  int64_t length = input.size(2);
  int64_t out_channels = weight.size(1);
  int64_t kernel_size = weight.size(2);

  TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
  TORCH_CHECK(bias.size(0) == out_channels, "bias size mismatch");
  TORCH_CHECK(stride > 0, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");
  TORCH_CHECK(output_padding >= 0 && output_padding < stride, "output_padding must be in [0, stride)");

  int64_t output_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
  TORCH_CHECK(output_length > 0, "invalid output length");

  // Create output tensor
  auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

  // Handle non-contiguous tensors by making contiguous copies if needed
  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  // bias is 1D, usually contiguous; ensure contiguous anyway
  auto bias_contig = bias.contiguous();

  cudnnHandle_t handle = get_cudnn_handle();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDNN(cudnnSetStream(handle, stream));

  // Create tensor descriptors
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

  // Set input descriptor: [batch, in_channels, length]
  // cuDNN typically uses NCHW layout; for 1D we use N, C, 1, L
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        in_channels,
                                        1,
                                        length));

  // Set output descriptor: [batch, out_channels, output_length]
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        out_channels,
                                        1,
                                        output_length));

  // Set filter descriptor: [in_channels, out_channels, kernel_size]
  // cuDNN filter layout for 1D is usually (out_channels, in_channels, kernel) but
  // PyTorch weight shape is (in_channels, out_channels, kernel).
  // We need to set the filter descriptor as (out_channels, in_channels, kernel).
  // cuDNN filter descriptor dimensions: [output_channels, input_channels, filter_height, filter_width]
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        out_channels,
                                        in_channels,
                                        1,
                                        kernel_size));

  // Set convolution descriptor
  // For ConvTranspose, use CUDNN_CROSS_CORRELATION and padding mode
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                              padding, 0,  // vertical pad, horizontal pad (padding top/bottom?)
                                              stride, 1,   // vertical stride, horizontal stride
                                              1, 1,        // dilation
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

  // cuDNN's convolution backward data corresponds to transposed convolution.
  // We'll use cudnnConvolutionBackwardData.
  // Need to set math type if needed
  cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH);

  // Determine algorithm
  cudnnConvolutionBwdDataAlgo_t algo;
  int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf_results[1];

  // We need to allocate a workspace? We'll use the algo selection to get workspace size.
  CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
      handle,
      filter_desc,
      output_desc,  // Note: In cudnnConvolutionBackwardData, the second arg is the diff output descriptor (input to backward)
      conv_desc,
      input_desc,   // diff input descriptor (output of backward)
      1,
      &returned_algo_count,
      perf_results));
  if (returned_algo_count == 0) {
    throw std::runtime_error("cudnnFindConvolutionBackwardDataAlgorithm returned no algorithm");
  }
  algo = perf_results[0].algo;

  // Get workspace size
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle,
      filter_desc,
      output_desc,
      conv_desc,
      input_desc,
      algo,
      &workspace_size));

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("cudaMalloc workspace failed");
    }
  }

  float alpha = 1.0f, beta = 0.0f;

  // Execute backward data: output = conv_transpose(input)
  CHECK_CUDNN(cudnnConvolutionBackwardData(
      handle,
      &alpha,
      filter_desc,
      weight_contig.data_ptr<float>(),
      output_desc,
      input_contig.data_ptr<float>(),
      conv_desc,
      algo,
      workspace,
      workspace_size,
      &beta,
      input_desc,
      output.data_ptr<float>()));

  // Add bias
  if (bias_contig.numel() > 0) {
    // We'll add bias using a simple kernel? But we are in .cpp, no kernels allowed.
    // We can use cudnnAddTensor or manually on host? No, must be on device.
    // cudnnAddTensor is allowed: it's a cuDNN operation.
    cudnnTensorDescriptor_t bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    // bias is [out_channels], we need to broadcast to [batch, out_channels, 1, output_length]
    // We can set bias as a tensor of shape [1, out_channels, 1, 1] and use cudnnAddTensor with beta=1.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          out_channels,
                                          1,
                                          1));
    float alpha_bias = 1.0f;
    float beta_bias = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(handle,
                               &alpha_bias,
                               bias_desc,
                               bias_contig.data_ptr<float>(),
                               &beta_bias,
                               output_desc,
                               output.data_ptr<float>()));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
  }

  // Cleanup
  if (workspace) cudaFree(workspace);
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));

  return output;
}

torch::Tensor addcmul_forward_wrapper(
    torch::Tensor input,
    torch::Tensor tensor1,
    torch::Tensor tensor2,
    double value) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(tensor1.is_cuda(), "tensor1 must be CUDA tensor");
  TORCH_CHECK(tensor2.is_cuda(), "tensor2 must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(tensor1.dtype() == torch::kFloat32, "tensor1 must be float32");
  TORCH_CHECK(tensor2.dtype() == torch::kFloat32, "tensor2 must be float32");

  // Check shapes match for elementwise operation
  TORCH_CHECK(input.sizes() == tensor1.sizes() && input.sizes() == tensor2.sizes(),
              "all tensors must have same shape");

  auto output = torch::empty_like(input);
  int64_t numel = input.numel();
  if (numel == 0) return output;

  // Ensure contiguous
  auto input_contig = input.contiguous();
  auto tensor1_contig = tensor1.contiguous();
  auto tensor2_contig = tensor2.contiguous();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  addcmul_forward(
      input_contig.data_ptr<float>(),
      tensor1_contig.data_ptr<float>(),
      tensor2_contig.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<float>(value),
      numel,
      stream);

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error after addcmul launch: " + std::string(cudaGetErrorString(err)));
  }

  return output;
}

void register_custom_ops(pybind11::module& m) {
  m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
        "ConvTranspose1d forward using cuDNN",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride") = 2,
        pybind11::arg("padding") = 1,
        pybind11::arg("output_padding") = 1);

  m.def("addcmul_forward", &addcmul_forward_wrapper,
        "addcmul forward",
        pybind11::arg("input"),
        pybind11::arg("tensor1"),
        pybind11::arg("tensor2"),
        pybind11::arg("value") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
