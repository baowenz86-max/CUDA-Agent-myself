#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
    cudaError_t flatten_launcher(
        const float* input,
        float* output,
        long long total_elements,
        cudaStream_t stream);

    cudaError_t weight_transpose_launcher(
        const float* src,
        float* dst,
        int in_channels,
        int out_channels,
        int kernel_size,
        cudaStream_t stream);

    cudaError_t copy_and_zero_pad_launcher(
        const float* src,
        float* dst,
        int total_rows,
        int L_out_no_pad,
        int L_out,
        cudaStream_t stream);

    cudaError_t add_bias_launcher(
        float* data,
        const float* bias,
        int batch,
        int out_channels,
        int L_out,
        cudaStream_t stream);
}

// cuDNN-based ConvTranspose1d launcher (implemented in .cpp to avoid cudnn.h in .cu)
static cudaError_t conv_transpose1d_forward_launcher(
    const float* x,
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
    cudaStream_t stream) {

  cudnnHandle_t cudnn_handle;
  cudnnStatus_t status = cudnnCreate(&cudnn_handle);
  if (status != CUDNN_STATUS_SUCCESS) return cudaErrorUnknown;
  status = cudnnSetStream(cudnn_handle, stream);
  if (status != CUDNN_STATUS_SUCCESS) {
    cudnnDestroy(cudnn_handle);
    return cudaErrorUnknown;
  }

  // Tensor descriptors
  cudnnTensorDescriptor_t x_desc, y_desc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnCreateTensorDescriptor(&y_desc);

  // x_desc: [batch, in_channels, 1, length]
  cudnnSetTensor4dDescriptor(x_desc,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             batch, in_channels, 1, length);

  // For transposed conv, we use cudnnConvolutionBackwardData.
  // The input to transposed conv (x) is the "dy" in backward data, and the output (y) is the "dx".
  // We compute without output_padding first into a temp buffer of size L_out_no_pad, then pad.
  int L_out_no_pad = L_out - output_padding;

  // y_desc for temp output: [batch, out_channels, 1, L_out_no_pad]
  cudnnSetTensor4dDescriptor(y_desc,
                             CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT,
                             batch, out_channels, 1, L_out_no_pad);

  // Filter descriptor: [out_channels, in_channels, 1, kernel_size]
  cudnnFilterDescriptor_t filter_desc;
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnSetFilter4dDescriptor(filter_desc,
                             CUDNN_DATA_FLOAT,
                             CUDNN_TENSOR_NCHW,
                             out_channels, in_channels, 1, kernel_size);

  // Convolution descriptor: same stride/padding/dilation as forward transposed conv
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnSetConvolution2dDescriptor(conv_desc,
                                  padding, padding,
                                  stride, stride,
                                  1, 1, // dilation
                                  CUDNN_CROSS_CORRELATION,
                                  CUDNN_DATA_FLOAT);

  // Transpose weight to cuDNN layout
  float* transposed_weight = nullptr;
  cudaError_t cuda_status = cudaMalloc(&transposed_weight,
                                       in_channels * out_channels * kernel_size * sizeof(float));
  if (cuda_status != cudaSuccess) {
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    return cuda_status;
  }

  cuda_status = weight_transpose_launcher(weight, transposed_weight,
                                          in_channels, out_channels, kernel_size, stream);
  if (cuda_status != cudaSuccess) {
    cudaFree(transposed_weight);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    return cuda_status;
  }

  // Allocate temp output if output_padding > 0, else use output directly
  float* temp_output = nullptr;
  if (output_padding > 0) {
    cuda_status = cudaMalloc(&temp_output, batch * out_channels * L_out_no_pad * sizeof(float));
    if (cuda_status != cudaSuccess) {
      cudaFree(transposed_weight);
      cudnnDestroyTensorDescriptor(x_desc);
      cudnnDestroyTensorDescriptor(y_desc);
      cudnnDestroyFilterDescriptor(filter_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
      cudnnDestroy(cudnn_handle);
      return cuda_status;
    }
  } else {
    temp_output = output; // use output directly
  }

  // Determine algorithm
  cudnnConvolutionBwdDataAlgo_t algo;
  status = cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle,
                                                    filter_desc,
                                                    x_desc, // dyDesc = x (small)
                                                    conv_desc,
                                                    y_desc, // dxDesc = temp_output (large)
                                                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                    0,
                                                    &algo);
  if (status != CUDNN_STATUS_SUCCESS) {
    if (output_padding > 0) cudaFree(temp_output);
    cudaFree(transposed_weight);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    return cudaErrorUnknown;
  }

  // Workspace
  size_t workspace_size = 0;
  status = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                                        filter_desc,
                                                        x_desc,
                                                        conv_desc,
                                                        y_desc,
                                                        algo,
                                                        &workspace_size);
  if (status != CUDNN_STATUS_SUCCESS) {
    if (output_padding > 0) cudaFree(temp_output);
    cudaFree(transposed_weight);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    return cudaErrorUnknown;
  }

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cuda_status = cudaMalloc(&workspace, workspace_size);
    if (cuda_status != cudaSuccess) {
      if (output_padding > 0) cudaFree(temp_output);
      cudaFree(transposed_weight);
      cudnnDestroyTensorDescriptor(x_desc);
      cudnnDestroyTensorDescriptor(y_desc);
      cudnnDestroyFilterDescriptor(filter_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
      cudnnDestroy(cudnn_handle);
      return cuda_status;
    }
  }

  // Perform backward data operation
  float alpha = 1.0f, beta = 0.0f;
  status = cudnnConvolutionBackwardData(cudnn_handle,
                                        &alpha,
                                        filter_desc,
                                        transposed_weight,
                                        x_desc,
                                        x, // dy
                                        conv_desc,
                                        algo,
                                        workspace,
                                        workspace_size,
                                        &beta,
                                        y_desc,
                                        temp_output); // dx

  if (status != CUDNN_STATUS_SUCCESS) {
    if (workspace) cudaFree(workspace);
    if (output_padding > 0) cudaFree(temp_output);
    cudaFree(transposed_weight);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    return cudaErrorUnknown;
  }

  // If output_padding > 0, copy temp_output to output and zero pad
  if (output_padding > 0) {
    cuda_status = copy_and_zero_pad_launcher(temp_output, output,
                                             batch * out_channels,
                                             L_out_no_pad, L_out, stream);
    if (cuda_status != cudaSuccess) {
      if (workspace) cudaFree(workspace);
      cudaFree(temp_output);
      cudaFree(transposed_weight);
      cudnnDestroyTensorDescriptor(x_desc);
      cudnnDestroyTensorDescriptor(y_desc);
      cudnnDestroyFilterDescriptor(filter_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
      cudnnDestroy(cudnn_handle);
      return cuda_status;
    }
    cudaFree(temp_output);
  }

  // Add bias if provided
  if (bias != nullptr) {
    cuda_status = add_bias_launcher(output, bias, batch, out_channels, L_out, stream);
    if (cuda_status != cudaSuccess) {
      if (workspace) cudaFree(workspace);
      cudaFree(transposed_weight);
      cudnnDestroyTensorDescriptor(x_desc);
      cudnnDestroyTensorDescriptor(y_desc);
      cudnnDestroyFilterDescriptor(filter_desc);
      cudnnDestroyConvolutionDescriptor(conv_desc);
      cudnnDestroy(cudnn_handle);
      return cuda_status;
    }
  }

  // Cleanup
  if (workspace) cudaFree(workspace);
  cudaFree(transposed_weight);
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn_handle);

  return cudaGetLastError();
}

// PyTorch wrapper for ConvTranspose1d
at::Tensor conv_transpose1d_forward(
    at::Tensor x,
    at::Tensor weight,
    std::optional<at::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, in_channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [in_channels, out_channels, kernel_size]");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    // Bias check
    at::Tensor bias_tensor;
    bool has_bias = bias.has_value();
    if (has_bias) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D [out_channels]");
        TORCH_CHECK(bias_tensor.size(0) == out_channels, "bias size must match out_channels");
    }

    // Compute output length
    int L_out = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "L_out must be positive");

    // Allocate output
    auto output = at::empty({batch, out_channels, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    const float* bias_ptr = has_bias ? bias_tensor.data_ptr<float>() : nullptr;
    cudaError_t err = conv_transpose1d_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        length,
        L_out,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "conv_transpose1d_forward_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for flatten
at::Tensor flatten(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");

    long long total_elements = x.numel();
    auto output = at::empty({x.size(0), total_elements / x.size(0)}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = flatten_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "flatten_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("flatten", &flatten,
          "Flatten tensor to 2D (batch, -1)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
