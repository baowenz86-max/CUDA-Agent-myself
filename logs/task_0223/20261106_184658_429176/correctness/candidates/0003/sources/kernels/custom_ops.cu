#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>

// Helper macro for cuDNN error checking
#define CHECK_CUDNN(expr)                                                     \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__,      \
              cudnnGetErrorString(status));                                    \
      return;                                                                  \
    }                                                                          \
  } while (0)

// Kernel to transpose weight from [C_in, C_out, K] to [C_out, C_in, K]
__global__ void transpose_weight_kernel(
    const float* __restrict__ weight_in,
    float* __restrict__ weight_out,
    int in_channels,
    int out_channels,
    int kernel_size)
{
    int total = in_channels * out_channels * kernel_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int k = idx % kernel_size;
    int tmp = idx / kernel_size;
    int c_out_orig = tmp % out_channels;  // original layout: [in][out][k]
    int c_in_orig = tmp / out_channels;
    // Transposed layout: [out][in][k]
    int out_idx = (c_out_orig * in_channels + c_in_orig) * kernel_size + k;
    weight_out[out_idx] = weight_in[idx];
}

// Launcher for custom_conv_transpose1d using cuDNN
// All inputs are raw pointers, output is preallocated by the binding (with zero padding if output_padding > 0)
// bias may be nullptr
// stream is the current CUDA stream
extern "C" void custom_conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out_base,  // without output_padding
    int length_out,       // with output_padding
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 || length_in <= 0 ||
        length_out_base <= 0 || length_out <= 0 || kernel_size <= 0 || stride <= 0 ||
        padding < 0 || output_padding < 0) {
        return;
    }

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t dy_desc, dx_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dy_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dx_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // dy is input x: [N, C_in, L_in, 1]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch, in_channels, length_in, 1));
    // dx is base output: [N, C_out, L_out_base, 1]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch, out_channels, length_out_base, 1));
    // bias descriptor: [1, C_out, 1, 1]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          1, out_channels, 1, 1));

    // filter descriptor: expected [C_out, C_in, kernel_size, 1], but weight is [C_in, C_out, kernel_size]
    // We will create a transposed copy on device using a kernel
    size_t weight_size = in_channels * out_channels * kernel_size * sizeof(float);
    float* weight_transposed = nullptr;
    cudaError_t cuda_err = cudaMalloc(&weight_transposed, weight_size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for weight_transposed failed\n");
        // cleanup and return
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Launch transpose kernel
    int total_weight = in_channels * out_channels * kernel_size;
    int block_size = 256;
    int blocks = (total_weight + block_size - 1) / block_size;
    transpose_weight_kernel<<<blocks, block_size, 0, stream>>>(
        weight, weight_transposed, in_channels, out_channels, kernel_size);
    // Check for launch error
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "transpose_weight_kernel launch failed: %s\n", cudaGetErrorString(cuda_err));
        cudaFree(weight_transposed);
        // cleanup descriptors, handle
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Now set filter descriptor on transposed weight: [C_out, C_in, kernel_size, 1]
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_size, 1));

    // Convolution descriptor: for backward data, parameters are for the forward convolution
    // that would have produced dx from dy. For ConvTranspose1d, forward conv has stride=stride,
    // padding=padding, dilation=1.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, 0, // pad_h, pad_w (only H dimension used)
                                                stride, 1,  // stride_h, stride_w
                                                1, 1,      // dilation_h, dilation_w
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Get algorithm
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &bwd_data_algo));

    // Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        bwd_data_algo,
        &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cuda_err = cudaMalloc(&workspace, workspace_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc workspace failed\n");
            cudaFree(weight_transposed);
            cudnnDestroyTensorDescriptor(dy_desc);
            cudnnDestroyTensorDescriptor(dx_desc);
            cudnnDestroyTensorDescriptor(bias_desc);
            cudnnDestroyFilterDescriptor(filter_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(cudnn_handle);
            return;
        }
    }

    // Allocate temporary base output (4D) on device
    size_t base_output_size = (size_t)batch * out_channels * length_out_base * sizeof(float);
    float* base_output = nullptr;
    cuda_err = cudaMalloc(&base_output, base_output_size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc base_output failed\n");
        if (workspace) cudaFree(workspace);
        cudaFree(weight_transposed);
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Perform backward data (transposed convolution)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_transposed,
        dy_desc,
        x,
        conv_desc,
        bwd_data_algo,
        workspace,
        workspace_size,
        &beta,
        dx_desc,
        base_output));

    // Add bias if present
    if (bias != nullptr) {
        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                                   &alpha_bias,
                                   bias_desc,
                                   bias,
                                   &beta_bias,
                                   dx_desc,
                                   base_output));
    }

    // Copy base_output to final output with optional zero padding at end
    // The final output tensor (preallocated by binding) has shape [batch, out_channels, length_out]
    // and is contiguous. We need to copy base_output segments of length length_out_base
    // to the first part of each channel segment, leaving zeros (binding should zero initialize).
    if (output_padding == 0) {
        // no padding, just copy the whole buffer
        cudaMemcpyAsync(output, base_output, base_output_size, cudaMemcpyDeviceToDevice, stream);
    } else {
        // Copy each batch-channel slice
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                size_t src_offset = (size_t)(b * out_channels + c) * length_out_base;
                size_t dst_offset = (size_t)(b * out_channels + c) * length_out;
                cudaMemcpyAsync(output + dst_offset,
                                base_output + src_offset,
                                length_out_base * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream);
            }
        }
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudaFree(base_output);
    cudaFree(weight_transposed);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
}

// Elementwise minimum with broadcasting over channel dimension.
__global__ void minimum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long y_channel_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride)
{
    long long total = batch * channels * length;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long l = idx % length;
        long long tmp = idx / length;
        long long c = tmp % channels;
        long long b = tmp / channels;

        const float x_val = x[b * x_batch_stride + c * x_channel_stride + l * x_length_stride];
        const float y_val = y[c * y_channel_stride];
        out[b * out_batch_stride + c * out_channel_stride + l * out_length_stride] =
            fminf(x_val, y_val);
    }
}

// Elementwise floor.
__global__ void floor_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride)
{
    long long total = batch * channels * length;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long l = idx % length;
        long long tmp = idx / length;
        long long c = tmp % channels;
        long long b = tmp / channels;

        const float x_val = x[b * x_batch_stride + c * x_channel_stride + l * x_length_stride];
        out[b * out_batch_stride + c * out_channel_stride + l * out_length_stride] = floorf(x_val);
    }
}

// Launcher for custom_minimum
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
    cudaStream_t stream)
{
    long long total = batch * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    long long blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    minimum_kernel<<<(int)blocks, block_size, 0, stream>>>(
        x, y, out,
        batch, channels, length,
        x_batch_stride, x_channel_stride, x_length_stride,
        y_channel_stride,
        out_batch_stride, out_channel_stride, out_length_stride);
}

// Launcher for custom_floor
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
    cudaStream_t stream)
{
    long long total = batch * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    long long blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    floor_kernel<<<(int)blocks, block_size, 0, stream>>>(
        x, out,
        batch, channels, length,
        x_batch_stride, x_channel_stride, x_length_stride,
        out_batch_stride, out_channel_stride, out_length_stride);
}
