#include <cuda_runtime.h>
#include <cudnn.h>

// Launcher declarations for binding layer (extern "C")
extern "C" void conv_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void upsample_nearest_forward(
    const float* input,
    float* output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    cudaStream_t stream);

extern "C" void one_hot_forward(
    const int64_t* input,
    int64_t* output,
    int total_elements,
    int num_classes,
    cudaStream_t stream);

// CUDA kernel for nearest neighbor upsample
__global__ void upsample_nearest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    int H_out,
    int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % channels;
    int b = tmp / channels;

    int h_in = h_out / scale_factor;
    int w_in = w_out / scale_factor;

    int input_idx = ((b * channels + c) * H_in + h_in) * W_in + w_in;
    output[idx] = input[input_idx];
}

// CUDA kernel for one-hot encoding
__global__ void one_hot_kernel(
    const int64_t* __restrict__ input,
    int64_t* __restrict__ output,
    int total_elements,
    int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t label = input[idx];
    int64_t* row = output + idx * num_classes;
    for (int k = 0; k < num_classes; ++k) {
        row[k] = (k == label) ? 1 : 0;
    }
}

// Launcher for conv_transpose using cuDNN
extern "C" void conv_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    // Compute output dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Initialize cuDNN if not already done (use static handle for performance)
    static cudnnHandle_t cudnn_handle = nullptr;
    static bool handle_initialized = false;
    if (!handle_initialized) {
        cudnnCreate(&cudnn_handle);
        cudnnSetStream(cudnn_handle, stream);
        handle_initialized = true;
    } else {
        cudnnSetStream(cudnn_handle, stream);
    }

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors for NCHW layout
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, in_channels, H_in, W_in);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, out_channels, H_out, W_out);

    // Set filter descriptor: ConvTranspose weight shape is [in_channels, out_channels, kernel_size, kernel_size]
    // cuDNN expects [out_channels, in_channels, kernel_size, kernel_size] for forward convolution,
    // but for backward data (transposed conv) it expects [in_channels, out_channels, kernel_size, kernel_size].
    // We'll use cudnnConvolutionBackwardData which treats weight as [in_channels, out_channels, kH, kW].
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               in_channels, out_channels, kernel_size, kernel_size);

    // Set convolution descriptor
    cudnnSetConvolution2dDescriptor(conv_desc,
                                     padding, padding,
                                     stride, stride,
                                     1, 1,  // dilation
                                     CUDNN_CROSS_CORRELATION,  // or CUDNN_CONVOLUTION; PyTorch uses cross-correlation
                                     CUDNN_DATA_FLOAT);

    // Set algorithm
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1; // Safe fallback
    // Optional: choose best algo with cudnnGetConvolutionBackwardDataAlgorithm, but for simplicity use default.

    // Allocate workspace if needed (none for this algo typically)
    size_t workspace_size = 0;
    void* workspace = nullptr;

    // Perform backward data convolution (transposed convolution)
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBackwardData(cudnn_handle,
                                 &alpha,
                                 filter_desc, weight,
                                 input_desc, input,
                                 conv_desc,
                                 algo,
                                 workspace, workspace_size,
                                 &beta,
                                 output_desc, output);

    // Add bias
    if (bias != nullptr) {
        // Use a simple kernel to add bias to each channel
        // We can launch a kernel that adds bias to each output element based on channel
        // Or use cudnnAddTensor, but simpler to use a custom kernel or cudaMemcpy2D pattern.
        // We'll implement a small kernel inline here.
        // For simplicity, we'll use cudaMemset for each channel? Better to handle in binding or separate kernel.
        // Since the contract requires bias addition, we can launch a custom kernel directly here.
        // But the requirement says no torch tensor in .cu, but raw CUDA is fine.
        // We'll define a local kernel for bias add.
        // However, to keep this file only with CUDA kernels and launchers, we'll define a bias add kernel above.
        // It will be a simple elementwise kernel. But we need to ensure it's called correctly.
        int total_out = batch * out_channels * H_out * W_out;
        int threads = 256;
        int blocks = (total_out + threads - 1) / threads;
        // Launch bias add kernel (need definition). We'll define it as a global function.
        // Instead of defining here, we can define a small inline device lambda? Not allowed in __global__.
        // So we need a separate __global__ function. We'll add it outside or use a helper.
        // For now, we'll define a kernel: add_bias_kernel(output, bias, batch, out_channels, H_out, W_out)
        // But we haven't declared it. We can add it above in the file.
        // To avoid complicating, we can add bias after cuDNN call using cudnnAddTensor or simply cudaMemcpy2D per channel.
        // Since bias addition is straightforward, we can use cudnnAddTensor with bias descriptor.
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, out_channels, 1, 1);
        cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias, &alpha, output_desc, output);
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    // Check for CUDA errors (optional but good)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real launcher we might want to handle error, but we can't throw here.
        // We'll just ignore or maybe print? Since this is a C interface, we can't throw.
        // Binding should check error after call.
        // We'll leave as is.
    }
}

// Launcher for upsample_nearest
extern "C" void upsample_nearest_forward(
    const float* input,
    float* output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    cudaStream_t stream)
{
    int H_out = H_in * scale_factor;
    int W_out = W_in * scale_factor;
    int total = batch * channels * H_out * W_out;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, H_in, W_in, scale_factor, H_out, W_out);
}

// Launcher for one_hot
extern "C" void one_hot_forward(
    const int64_t* input,
    int64_t* output,
    int total_elements,
    int num_classes,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, num_classes);
}
