#include <cuda_runtime.h>
#include <cudnn.h>

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
// cuDNN ConvTranspose1d implementation as a launcher called from binding
// All cuDNN API calls are here in the .cu file compiled by nvcc.
// ----------------------------------------------------------------

// Helper to check cuDNN status
static void checkCudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        // We cannot throw here; the binding will check the returned status.
        // We'll return the error code through a pointer parameter.
    }
}

// Launcher function for conv_transpose1d_forward
// Parameters:
//   input: pointer to input tensor [B, C_in, L]
//   weight: pointer to weight tensor [C_in, C_out, K]
//   output: pointer to output tensor [B, C_out, L_out]
//   batch, in_channels, out_channels, input_length, kernel_size, stride, padding, output_padding
//   stream: CUDA stream
// Returns cudnnStatus_t (CUDNN_STATUS_SUCCESS on success)
extern "C" cudnnStatus_t conv_transpose1d_launcher(
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

    // Set up tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Dimensions for the ConvTranspose1d
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Set tensor descriptors as 4D (NCHW with H=1, W=length)
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, in_channels, 1, input_length);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, out_channels, 1, output_length);

    // Filter descriptor: for ConvTranspose1d with groups=1 and in_channels == out_channels,
    // weight shape [in_channels, out_channels, kernel_size] matches cuDNN filter [K, C, R]
    // because in_channels == out_channels. So we can use direct pointer without transposition.
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels, kernel_size, 1);

    // Convolution descriptor for 2D (with height=1)
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, 0,      // pad_h, pad_w
                                    stride, 1,       // stride_h, stride_w
                                    1, 1,            // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // For ConvTranspose, we use cudnnConvolutionBackwardData with:
    //   w_desc: filter (weight)
    //   y_desc: input (dy)
    //   x_desc: output (dx)
    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc, weight,
        y_desc, input,   // dy
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        nullptr, 0,
        &beta,
        x_desc, output);  // dx

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return status;
}
