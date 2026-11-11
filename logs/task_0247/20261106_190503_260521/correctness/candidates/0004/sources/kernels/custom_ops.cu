#include <cuda_runtime.h>
#include <cudnn.h>

// Error checking macro for cuDNN and CUDA (only used in launchers)
#define CUDNN_CHECK(expr)                                                  \
  do {                                                                      \
    cudnnStatus_t status = (expr);                                         \
    if (status != CUDNN_STATUS_SUCCESS) {                                  \
      printf("cuDNN error at %s:%d: %s\n", __FILE__, __LINE__,            \
             cudnnGetErrorString(status));                                  \
      return;                                                               \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr)                                                  \
  do {                                                                      \
    cudaError_t err = (expr);                                              \
    if (err != cudaSuccess) {                                              \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
             cudaGetErrorString(err));                                      \
      return;                                                               \
    }                                                                        \
  } while (0)

// Helper: create tensor descriptor for NCHW with L as width, H=1
extern "C" cudnnTensorDescriptor_t create_tensor_desc(int n, int c, int l, int stride_n, int stride_c, int stride_l) {
  cudnnTensorDescriptor_t desc;
  cudnnCreateTensorDescriptor(&desc);
  int dims[4] = {n, c, 1, l};
  int strides[4] = {stride_n, stride_c, 1, stride_l};
  cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, dims, strides);
  return desc;
}

// Helper: create filter descriptor for 4D filter (K, C, 1, R)
extern "C" cudnnFilterDescriptor_t create_filter_desc(int k, int c, int r) {
  cudnnFilterDescriptor_t desc;
  cudnnCreateFilterDescriptor(&desc);
  int dims[4] = {k, c, 1, r};
  cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, dims);
  return desc;
}

// Helper: create convolution descriptor
extern "C" cudnnConvolutionDescriptor_t create_conv_desc(int stride, int padding, int output_padding) {
  cudnnConvolutionDescriptor_t desc;
  cudnnCreateConvolutionDescriptor(&desc);
  int pad[2] = {0, padding};
  int stride_[2] = {1, stride};
  int dilation[2] = {1, 1};
  cudnnSetConvolution2dDescriptor(desc,
                                  pad[0], pad[1],
                                  stride_[0], stride_[1],
                                  dilation[0], dilation[1],
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  return desc;
}

// ConvTranspose1d launcher: input (N, C_in, L_in), weight (C_in, C_out, K), output (N, C_out, L_out)
// Uses cudnnConvolutionBackwardData
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream) {

  // Compute output length
  int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;

  cudnnHandle_t handle;
  cudnnCreate(&handle);
  cudnnSetStream(handle, stream);

  // Descriptors
  cudnnTensorDescriptor_t input_desc = create_tensor_desc(N, C_in, L_in, C_in * L_in, L_in, 1);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(N, C_out, L_out, C_out * L_out, L_out, 1);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(C_in, C_out, K);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(stride, padding);

  float alpha = 1.0f, beta = 0.0f;
  cudnnStatus_t status = cudnnConvolutionBackwardData(
      handle,
      &alpha,
      filter_desc, weight,
      input_desc, input,
      conv_desc,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      nullptr, 0,
      &beta,
      output_desc, output);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("conv_transpose cudnnConvolutionBackwardData failed: %s\n", cudnnGetErrorString(status));
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

// Conv1d launcher: input (N, C_in, L_in), weight (C_out, C_in, K), bias (C_out) optional, output (N, C_out, L_out)
extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding,
    cudaStream_t stream) {

  int L_out = (L_in + 2 * padding - K) / stride + 1;

  cudnnHandle_t handle;
  cudnnCreate(&handle);
  cudnnSetStream(handle, stream);

  cudnnTensorDescriptor_t input_desc = create_tensor_desc(N, C_in, L_in, C_in * L_in, L_in, 1);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(N, C_out, L_out, C_out * L_out, L_out, 1);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(C_out, C_in, K);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(stride, padding);

  float alpha = 1.0f, beta = 0.0f;
  cudnnStatus_t status = cudnnConvolutionForward(
      handle,
      &alpha,
      input_desc, input,
      filter_desc, weight,
      conv_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      nullptr, 0,
      &beta,
      output_desc, output);
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("conv1d cudnnConvolutionForward failed: %s\n", cudnnGetErrorString(status));
  }

  // Add bias if provided, using custom kernel
  if (bias != nullptr) {
    // Launch simple kernel to add per-channel bias
    // grid-stride loop, one thread per output element
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Need a kernel; we'll define it below
    // extern "C" __global__ void add_bias_kernel(float* output, const float* bias, int N, int C_out, int L_out)
    // We'll declare and use a kernel defined in this file:
    // (implementation below)
    // For now, launch it:
    add_bias_kernel<<<blocks, threads, 0, stream>>>(output, bias, N, C_out, L_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("add_bias_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

// CUDA kernel for bias addition: output[n,c,l] += bias[c]
__global__ void add_bias_kernel(float* output, const float* bias, int N, int C_out, int L_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C_out * L_out;
  if (idx < total) {
    int l = idx % L_out;
    int c = (idx / L_out) % C_out;
    output[idx] += bias[c];
  }
}
