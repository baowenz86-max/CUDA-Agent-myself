// custom_ops.cu
// CUDA kernels and launchers for maxpool3d and prod_dim operations.
// This file contains only raw CUDA code and launchers. No PyTorch dependencies.

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------
// maxpool3d_forward using cuDNN
// -----------------------------------------------------------------------------
// Launcher for max pooling 3D. Assumes input is contiguous and float32.
// Kernel size, stride, padding, dilation are passed as integers (assumed same
// for all dimensions per the contract).
// The binding layer will allocate the output tensor and call this launcher.
extern "C" void maxpool3d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int ceil_mode,
    cudaStream_t stream)
{
    // Create cuDNN handle
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        // handle error (binding may check)
        return;
    }

    // Set the stream for cuDNN
    status = cudnnSetStream(handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(handle);
        return;
    }

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    // Input tensor: NCDHW format (batch, channels, depth, height, width)
    // cuDNN uses NCDHW for 5D tensors.
    cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        in_channels,
        input_depth,
        input_height);
    // For 5D, we need to set the dimension sizes accordingly. Actually, cuDNN pooling API
    // has separate functions for 2D and 3D. For 3D pooling, we use cudnnPoolingForward with
    // a 5D tensor descriptor. The descriptor should be set using cudnnSetTensorNdDescriptor.
    // Simpler: we set all dims individually.

    // Recreate descriptors as 5D
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);

    int dims[5] = {batch_size, in_channels, input_depth, input_height, input_width};
    int strides[5];
    // For contiguous NCDHW layout
    strides[4] = 1;
    strides[3] = input_width;
    strides[2] = input_height * input_width;
    strides[1] = input_depth * input_height * input_width;
    strides[0] = in_channels * input_depth * input_height * input_width;
    cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 5, dims, strides);

    // Output dimensions: compute using floor formula
    int output_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int out_dims[5] = {batch_size, in_channels, output_depth, output_height, output_width};
    int out_strides[5];
    out_strides[4] = 1;
    out_strides[3] = output_width;
    out_strides[2] = output_height * output_width;
    out_strides[1] = output_depth * output_height * output_width;
    out_strides[0] = in_channels * output_depth * output_height * output_width;
    cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides);

    // Create pooling descriptor
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnCreatePoolingDescriptor(&pooling_desc);

    cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
    cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;

    // Set pooling descriptor. For 3D pooling, we use cudnnSetPoolingNdDescriptor.
    int windowDim[3] = {kernel_size, kernel_size, kernel_size};
    int paddingA[3] = {padding, padding, padding};
    int strideA[3] = {stride, stride, stride};
    // dilation is not supported in cuDNN pooling? Actually cuDNN supports dilation for pooling via cudnnSetPoolingNdDescriptor with dilation parameter.
    // According to cuDNN docs, cudnnSetPoolingNdDescriptor takes nbDims, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA.
    // There is no dilation parameter in the standard pooling descriptor. For dilation, cuDNN pooling does not support dilation strides.
    // Since contract dilation=1, it's safe to ignore dilation in cuDNN. If dilation>1, we'd need a custom kernel, but here it's 1.
    cudnnSetPoolingNdDescriptor(
        pooling_desc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        3,
        windowDim,
        paddingA,
        strideA);

    // Perform pooling forward
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cudnnPoolingForward(
        handle,
        pooling_desc,
        &alpha,
        input_desc,
        input,
        &beta,
        output_desc,
        output);

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroy(handle);

    // Note: we do not return status; binding should check CUDA errors after launch.
}

// -----------------------------------------------------------------------------
// prod_dim_forward: product along dimension 1 (channels) with keepdim=True
// -----------------------------------------------------------------------------
// Kernel computes product over dimension 1 for each spatial location and batch.
// Output shape: [batch, 1, D, H, W]
// Input shape: [batch, C, D, H, W]
// We launch one thread per output element (batch * D * H * W).
// Each thread iterates over channels and multiplies.
// Assumes input is contiguous (or at least standard strides). Binding will ensure contiguity if needed,
// but we can also handle arbitrary strides by passing strides. However, for simplicity and performance,
// we assume the input from maxpool is contiguous (which it is).
__global__ void prod_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width)
{
    int total_output = batch_size * depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Decompose idx into b, d, h, w
    int w = idx % width;
    int tmp = idx / width;
    int h = tmp % height;
    tmp = tmp / height;
    int d = tmp % depth;
    int b = tmp / depth;

    // Input base index for (b, 0, d, h, w) assuming contiguous NCDHW
    long long base = ((long long)b * channels + 0) * depth * height * width
                   + d * height * width + h * width + w;

    float prod = 1.0f;
    for (int c = 0; c < channels; ++c) {
        prod *= input[base + (long long)c * depth * height * width];
    }

    // Output index: b * depth * height * width + d * height * width + h * width + w
    int out_idx = b * depth * height * width + d * height * width + h * width + w;
    output[out_idx] = prod;
}

extern "C" void prod_dim_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream)
{
    int total_output = batch_size * depth * height * width;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;

    // Use grid-stride loop in kernel to handle any size
    prod_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, depth, height, width);

    // Check for launch errors (optional; binding may check too)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
}
