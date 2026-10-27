#include <cuda_runtime.h>
#include <cudnn.h>

// kernels/custom_ops.cu
//
// This file contains both the custom CUDA kernel (ge_forward) and the
// ConvTranspose1d implementation using cuDNN.
// The binding layer (custom_ops_binding.cpp) will call launchers from this file.

// ----------------------------------------------------------------
// ge_forward: element-wise input >= other (other is a scalar float)
// Input:  float32 tensor, arbitrary contiguous or non-contiguous layout
// Output: bool tensor, same shape as input
// The kernel uses a grid-stride loop to handle any number of elements.
// ----------------------------------------------------------------

__global__ void ge_kernel(const float* __restrict__ input,
                          bool* __restrict__ output,
                          int64_t numel,
                          float other) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] >= other);
    }
}

// Host launcher for ge_forward. The binding layer will call this function
// with raw pointers and the current CUDA stream.
extern "C" void ge_forward_launcher(const float* input,
                                    bool* output,
                                    int64_t numel,
                                    float other,
                                    cudaStream_t stream) {
    if (numel <= 0) {
        return;
    }
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding limits; grid-stride loop handles the rest.
    const int64_t max_grid = 65535;
    if (grid_size > max_grid) {
        grid_size = max_grid;
    }
    ge_kernel<<<static_cast<int>(grid_size), block_size, 0, stream>>>(
        input, output, numel, other);
}

// ----------------------------------------------------------------
// ConvTranspose1d implementation using cuDNN.
// Since we cannot easily include cudnn.h in the .cpp binding due to include path
// issues, we implement the cuDNN call here in the .cu file.
// The function below is a launcher that will be called from the binding.
// ----------------------------------------------------------------

// Helper to check cuDNN status (will be used internally)
void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        // In a .cu file we cannot use TORCH_CHECK, so we just print an error and abort.
        // The binding will also check the returned status and throw a proper error.
        printf("cuDNN error: %s: %s\n", msg, cudnnGetErrorString(status));
        // We cannot throw, but we can set an error flag. However, for simplicity,
        // we just return and let the binding check the status.
    }
}

// Function that performs the cuDNN convolution and returns status.
// This is called from the launcher.
cudnnStatus_t convTranspose1dCudnnImpl(
    const float* input,
    const float* weight,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream,
    cudnnHandle_t cudnn_handle) {

    // Set up tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Use 4D descriptors with H=1
    int x_dims_4d[4] = {batch, in_channels, 1, input_length};
    int y_dims_4d[4] = {batch, out_channels, 1, output_length};

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, in_channels, 1, input_length);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, out_channels, 1, output_length);

    // Filter descriptor: [out_channels, in_channels, kernel_size, 1]
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels, kernel_size, 1);

    // Convolution descriptor
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, 0,      // pad_h, pad_w
                                    stride, 1,       // stride_h, stride_w
                                    1, 1,            // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION, // mode
                                    CUDNN_DATA_FLOAT);

    // For ConvTranspose, use cudnnConvolutionBackwardData
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc, weight,
        y_desc, input,   // dy
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,  // could use ALGO_0
        nullptr, 0,
        &beta,
        x_desc, output);  // dx

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    return status;
}

// Launcher for conv_transpose1d_forward. It creates a cuDNN handle, sets stream,
// calls the implementation, and destroys the handle.
extern "C" cudnnStatus_t conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream) {

    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        return status;
    }
    cudnnSetStream(cudnn_handle, stream);

    status = convTranspose1dCudnnImpl(input, weight, output,
                                     batch, in_channels, out_channels,
                                     input_length, kernel_size, stride, padding, output_padding,
                                     stream, cudnn_handle);

    cudnnDestroy(cudnn_handle);
    return status;
}
