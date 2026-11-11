#include <cuda_runtime.h>
#include <cudnn.h>

// Error checking macros for CUDA and cuDNN (no torch dependency)
#define CUDNN_CHECK(expr)                                                      \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      const char* err = cudnnGetErrorString(status);                           \
      printf("cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, err);           \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      return;                                                                  \
    }                                                                          \
  } while (0)

// Helper: create 4D tensor descriptor for 1D tensor (N, C, L) -> (N, C, 1, L)
static cudnnTensorDescriptor_t create_tensor_desc(const float* data, int n, int c, int l) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  int dims[4] = {n, c, 1, l};
  int strides[4] = {c * l, l, 1, 1};
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, dims, strides));
  return desc;
}

// Helper: create filter descriptor for 1D conv (K, C, 1, R)
static cudnnFilterDescriptor_t create_filter_desc(const float* weight, int k, int c, int r) {
  cudnnFilterDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
  int dims[4] = {k, c, 1, r};
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, dims));
  return desc;
}

// Helper: create 2D convolution descriptor for 1D conv
static cudnnConvolutionDescriptor_t create_conv_desc(int stride, int padding) {
  cudnnConvolutionDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
  int pad[2] = {0, padding};
  int stride_[2] = {1, stride};
  int dilation[2] = {1, 1};
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc, pad[0], pad[1],
                                              stride_[0], stride_[1],
                                              dilation[0], dilation[1],
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  return desc;
}

// Compute output length for Conv or ConvTranspose
static int64_t compute_conv_out_length(int64_t in_len, int64_t kernel,
                                       int64_t stride, int64_t padding,
                                       bool transposed, int64_t output_padding) {
  if (transposed) {
    return (in_len - 1) * stride - 2 * padding + kernel + output_padding;
  } else {
    return (in_len - kernel) / stride + 1;  // padding=0 for conv1d in this task
  }
}

// CUDA kernel for add_bias (bias shape [C, 1] broadcast to [N, C, L])
__global__ void add_bias_kernel(float* output, const float* input, const float* bias,
                                int64_t total_elems, int64_t C, int64_t L) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elems) {
    int64_t c = (idx / L) % C;
    output[idx] = input[idx] + bias[c];
  }
}

// Launcher for add_bias (input and output same shape [N, C, L])
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias,
                                  int64_t N, int64_t C, int64_t L, cudaStream_t stream) {
  int64_t total = N * C * L;
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  add_bias_kernel<<<grid, block, 0, stream>>>(output, input, bias, total, C, L);
  CUDA_CHECK(cudaGetLastError());
}

// Launcher for ConvTranspose1d (input: [N, C_in, L_in], weight: [C_in, C_out, K])
// output: [N, C_out, L_out]
extern "C" void conv_transpose1d_launcher(
    const float* input, const float* weight, float* output,
    int64_t N, int64_t C_in, int64_t L_in, int64_t C_out, int64_t K,
    int64_t stride, int64_t padding, int64_t output_padding, cudaStream_t stream) {

  int64_t L_out = compute_conv_out_length(L_in, K, stride, padding, true, output_padding);

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, stream));

  cudnnTensorDescriptor_t input_desc = create_tensor_desc(input, (int)N, (int)C_in, (int)L_in);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(output, (int)N, (int)C_out, (int)L_out);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(weight, (int)C_in, (int)C_out, (int)K);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc((int)stride, (int)padding);

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

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

// Launcher for Conv1d (input: [N, C_in, L_in], weight: [C_out, C_in, K])
// output: [N, C_out, L_out], optional bias [C_out]
extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t N, int64_t C_in, int64_t L_in, int64_t C_out, int64_t K,
    int64_t stride, int64_t padding, cudaStream_t stream) {

  int64_t L_out = compute_conv_out_length(L_in, K, stride, padding, false, 0);

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, stream));

  cudnnTensorDescriptor_t input_desc = create_tensor_desc(input, (int)N, (int)C_in, (int)L_in);
  cudnnTensorDescriptor_t output_desc = create_tensor_desc(output, (int)N, (int)C_out, (int)L_out);
  cudnnFilterDescriptor_t filter_desc = create_filter_desc(weight, (int)C_out, (int)C_in, (int)K);
  cudnnConvolutionDescriptor_t conv_desc = create_conv_desc((int)stride, (int)padding);

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

  // Add bias if not null
  if (bias != nullptr) {
    int64_t total = N * C_out * L_out;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    add_bias_kernel<<<grid, block, 0, stream>>>(output, output, bias, total, C_out, L_out);
    CUDA_CHECK(cudaGetLastError());
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}
