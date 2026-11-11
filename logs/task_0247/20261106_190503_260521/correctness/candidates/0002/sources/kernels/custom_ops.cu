// CUDA source with cuDNN launchers and add_bias kernel
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>

// Error checking macro for cuDNN
#define CUDNN_CHECK(expr)                                                        \
  do {                                                                           \
    cudnnStatus_t status = (expr);                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                        \
      const char* err = cudnnGetErrorString(status);                             \
      printf("cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, err);          \
      return;                                                                    \
    }                                                                            \
  } while (0)

// Simple add bias kernel: output[n,c,l] = input[n,c,l] + bias[c]
__global__ void add_bias_kernel(float* output, const float* input, const float* bias,
                                int64_t N, int64_t C, int64_t L) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = N * C * L;
  if (idx < total) {
    int64_t l = idx % L;
    int64_t c = (idx / L) % C;
    output[idx] = input[idx] + bias[c];
  }
}

// cuDNN helper to create tensor descriptor for 1D conv (N, C, 1, L)
static cudnnTensorDescriptor_t create_tensor_desc(const float* data, int N, int C, int L) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  int dims[4] = {N, C, 1, L};
  int strides[4] = {C * L, L, 1, 1}; // assuming contiguous
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, dims, strides));
  return desc;
}

// cuDNN helper for filter descriptor
static cudnnFilterDescriptor_t create_filter_desc(const float* weight, int K, int C, int R) {
  cudnnFilterDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
  int dims[4] = {K, C, 1, R};
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, dims));
  return desc;
}

// cuDNN helper for convolution descriptor
static cudnnConvolutionDescriptor_t create_conv_desc(int stride, int padding) {
  cudnnConvolutionDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
  int pad[2] = {0, padding};
  int stride_[2] = {1, stride};
  int dilation[2] = {1, 1};
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc,
                                               pad[0], pad[1],
                                               stride_[0], stride_[1],
                                               dilation[0], dilation[1],
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  return desc;
}

// ConvTranspose1d launcher using cuDNN BackwardData
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream) {
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, stream));

  cudnnTensorDescriptor_t input_desc = create_tensor_desc(input, N, C_in, L_in);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(output, N, C_out, L_out);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(weight, C_in, C_out, kernel_size);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(stride, padding);

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionBackwardData(
      handle,
      &alpha,
      filter_desc, weight,
      input_desc, input,
      conv_desc,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      nullptr, 0,
      &beta,
      output_desc, output));

  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

// Conv1d launcher using cuDNN Forward
extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, bool has_bias,
    cudaStream_t stream) {
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, stream));

  cudnnTensorDescriptor_t input_desc = create_tensor_desc(input, N, C_in, L_in);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(output, N, C_out, L_out);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(weight, C_out, C_in, kernel_size);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc(stride, padding);

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      handle,
      &alpha,
      input_desc, input,
      filter_desc, weight,
      conv_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      nullptr, 0,
      &beta,
      output_desc, output));

  if (has_bias) {
    // Add bias using simple CUDA kernel
    int64_t total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(output, output, bias, N, C_out, L_out);
  }

  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

// Add bias launcher
extern "C" void add_bias_launcher(
    float* output, const float* input, const float* bias,
    int64_t N, int64_t C, int64_t L,
    cudaStream_t stream) {
  int64_t total = N * C * L;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads, 0, stream>>>(output, input, bias, N, C, L);
}
