#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void custom_pow_launcher(
    float* output,
    const float* input,
    const float exponent,
    int numel,
    int ndim,
    const int64_t* shape,
    const int64_t* strides,
    cudaStream_t stream);

// Utility macro for cuDNN error checking
#define CHECK_CUDNN(expr)                                                      \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      const char* err_str = cudnnGetErrorString(status);                       \
      throw std::runtime_error(std::string("cuDNN error: ") + err_str);        \
    }                                                                          \
  } while (0)

// Compute output length for ConvTranspose1d
static inline int64_t compute_output_length(int64_t input_length, int64_t kernel_size,
                                           int64_t stride, int64_t padding, int64_t output_padding) {
  return (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// custom_conv_transpose1d binding
// Args: input (CUDA float32), weight (CUDA float32), bias (CUDA float32, may be None),
//       stride (int), padding (int), output_padding (int)
// Returns: output tensor shape [N, out_channels, output_length]
static torch::Tensor custom_conv_transpose1d(torch::Tensor input, torch::Tensor weight,
                                             torch::Tensor bias, int64_t stride,
                                             int64_t padding, int64_t output_padding) {
  // Basic checks
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
  }
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  if (bias.defined()) {
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  }
  TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L)");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");
  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias.size(0) == weight.size(1), "bias size must match out_channels");
  }
  TORCH_CHECK(stride > 0, "stride must be positive");
  TORCH_CHECK(padding >= 0, "padding must be non-negative");
  TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

  // Ensure contiguous tensors for cuDNN API
  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  torch::Tensor bias_contig;
  if (bias.defined()) {
    bias_contig = bias.contiguous();
  }

  const int64_t N = input_contig.size(0);
  const int64_t C_in = input_contig.size(1);
  const int64_t L_in = input_contig.size(2);
  const int64_t C_out = weight_contig.size(1);
  const int64_t kernel_size = weight_contig.size(2);

  TORCH_CHECK(C_in == weight_contig.size(0), "input channels must match weight in_channels");
  const int64_t L_out = compute_output_length(L_in, kernel_size, stride, padding, output_padding);

  // Allocate output tensor (contiguous)
  auto output = torch::empty({N, C_out, L_out}, input_contig.options());

  // cuDNN handles and descriptors
  cudnnHandle_t cudnn_handle;
  CHECK_CUDNN(cudnnCreate(&cudnn_handle));

  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));

  // Set tensor descriptors: layout NCHW with L as the spatial dimension (W)
  // cuDNN expects 4D or 5D tensors; for 1D we treat it as 2D with height=1.
  // Use NCHW layout with H=1, W=L.
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         static_cast<int>(N), static_cast<int>(C_in),
                                         1, static_cast<int>(L_in)));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         static_cast<int>(N), static_cast<int>(C_out),
                                         1, static_cast<int>(L_out)));
  // Filter descriptor: for convolution (not transposed), shape is (C_out, C_in, kernel_size)
  // cuDNN filter format is (K, C, H, W) for NCHW; here K=C_out, C=C_in, H=1, W=kernel_size.
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                         static_cast<int>(C_out), static_cast<int>(C_in),
                                         1, static_cast<int>(kernel_size)));

  // Convolution descriptor for transposed convolution (backward data)
  cudnnConvolutionDescriptor_t conv_desc;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  // Padding for 2D: vertical padding = 0, horizontal padding = given padding
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                              0, static_cast<int>(padding),
                                              static_cast<int>(stride), static_cast<int>(stride),
                                              1, 1,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

  // Algorithm selection
  cudnnConvolutionBwdDataAlgo_t algo;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle,
      filter_desc,
      input_desc,
      conv_desc,
      output_desc,
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      0,
      &algo));

  // Workspace size
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle,
      filter_desc,
      input_desc,
      conv_desc,
      output_desc,
      algo,
      &workspace_size));

  // Allocate workspace if needed
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
    TORCH_CHECK(cuda_status == cudaSuccess, "cudaMalloc failed for cuDNN workspace");
  }

  // Get current stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

  // Execute backward data convolution (transposed convolution)
  const float alpha = 1.0f;
  const float beta = 0.0f;  // we will add bias separately or use beta=1 if no bias? Actually we need bias addition.
  // We can either set beta=1 and initialize output with bias, or set beta=0 and add bias after.
  // Simpler: set beta=0, then add bias with a separate operation or use cudnnAddTensor.
  // Let's use beta=0 and then add bias if defined.
  CHECK_CUDNN(cudnnConvolutionBackwardData(
      cudnn_handle,
      &alpha,
      filter_desc,
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
    // Use cudnnAddTensor to add bias per channel.
    cudnnTensorDescriptor_t bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, static_cast<int>(C_out), 1, 1));
    const float beta_bias = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                               &alpha,
                               bias_desc,
                               bias_contig.data_ptr<float>(),
                               &beta_bias,
                               output_desc,
                               output.data_ptr<float>()));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
  }

  // Cleanup
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  CHECK_CUDNN(cudnnDestroy(cudnn_handle));

  if (workspace) {
    cudaFree(workspace);
  }

  // Check for CUDA errors
  cudaError_t cuda_err = cudaGetLastError();
  TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error after conv_transpose: ", cudaGetErrorString(cuda_err));

  return output;
}

// custom_pow binding
// Args: input tensor (CUDA float32), exponent (double)
// Returns: tensor of same shape as input
static torch::Tensor custom_pow(torch::Tensor input, double exponent) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

  // Ensure contiguous input for launcher simplicity (but launcher handles strides,
  // so we can pass non-contiguous directly if we also pass strides. However, output allocation
  // below will be contiguous and we need to pass output shape/strides.)
  // We'll make input contiguous to simplify but still pass shape and strides arrays.
  auto input_contig = input.contiguous();
  auto output = torch::empty_like(input_contig);

  const int numel = static_cast<int>(input_contig.numel());
  if (numel == 0) {
    return output;
  }

  const int ndim = static_cast<int>(input_contig.dim());
  std::vector<int64_t> shape(ndim);
  std::vector<int64_t> strides(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape[i] = input_contig.size(i);
    strides[i] = input_contig.stride(i);
  }

  // Get current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  custom_pow_launcher(
      output.data_ptr<float>(),
      input_contig.data_ptr<float>(),
      static_cast<float>(exponent),
      numel,
      ndim,
      shape.data(),
      strides.data(),
      stream);

  // Check for CUDA errors after launch
  cudaError_t cuda_err = cudaGetLastError();
  TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error after pow: ", cudaGetErrorString(cuda_err));

  return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
  m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
        "ConvTranspose1d using cuDNN",
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
  m.def("custom_pow", &custom_pow,
        "Element-wise power",
        py::arg("input"), py::arg("exponent"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
