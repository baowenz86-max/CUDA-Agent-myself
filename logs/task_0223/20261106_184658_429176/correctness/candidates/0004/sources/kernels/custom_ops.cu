#include <cuda_runtime.h>
#include <cudnn.h>

// -----------------------------------------------------------------------------
// Elementwise minimum with broadcasting over channel dimension.
// Input tensors: x has shape [batch, out_channels, length_out] (3D, strides arbitrary)
// y has shape [out_channels, 1] and is broadcast along batch and length dimensions.
// Output has same shape as x, float32.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Elementwise floor with support for arbitrary strides.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Launcher for custom_minimum
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Launcher for custom_floor
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Helper kernel: transpose weight [C_in, C_out, K] -> [C_out, C_in, K]
// Used because cuDNN's backward data filter expects [K, C_out, C_in] in NCHW,
// but we create a 4D filter descriptor as [C_out, C_in, K, 1] if we use
// CUDNN_TENSOR_NCHW. Actually PyTorch ConvTranspose1d weight is [in_channels,
// out_channels, kernel_size], so we need to produce filter in that layout.
// We will allocate transposed weight and fill it with a simple kernel.
// -----------------------------------------------------------------------------
__global__ void transpose_weight_kernel(
    const float* __restrict__ w_in,
    float* __restrict__ w_out,
    int in_channels,
    int out_channels,
    int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_channels * out_channels * kernel_size;
    if (idx >= total) return;
    int k = idx % kernel_size;
    int tmp = idx / kernel_size;
    int oc = tmp % out_channels;
    int ic = tmp / out_channels;
    // w_in is [in_channels, out_channels, kernel_size] contiguous
    // index = ic * (out_channels * kernel_size) + oc * kernel_size + k
    const float val = w_in[ic * out_channels * kernel_size + oc * kernel_size + k];
    // w_out is [out_channels, in_channels, kernel_size] contiguous
    // output index = oc * (in_channels * kernel_size) + ic * kernel_size + k
    w_out[oc * in_channels * kernel_size + ic * kernel_size + k] = val;
}

// -----------------------------------------------------------------------------
// cuDNN-based ConvTranspose1d launcher
// Inputs:
//   x: input tensor, shape [batch, in_channels, length_in], contiguous
//   weight: ConvTranspose1d weight, shape [in_channels, out_channels, kernel_size], contiguous
//   bias: optional, shape [out_channels], contiguous, can be nullptr if not defined
//   output: preallocated output tensor, shape [batch, out_channels, length_out], contiguous
//   batch, in_channels, out_channels, length_in, length_out, kernel_size, stride, padding, output_padding
//   stream: CUDA stream to use
// The launcher performs the full transposed convolution including bias and output_padding.
// -----------------------------------------------------------------------------
extern "C" void custom_conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    // Compute base output length (without output_padding)
    int length_out_base = length_out - output_padding;
    if (length_out_base <= 0) return; // error case should have been caught in binding

    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) return;
    cudnnSetStream(cudnn_handle, stream);

    // Descriptors for backward data (transposed convolution)
    // dy = input x: shape [batch, in_channels, length_in, 1]
    // dx = output base: shape [batch, out_channels, length_out_base, 1]
    // filter = transposed weight: shape [out_channels, in_channels, kernel_size, 1]
    cudnnTensorDescriptor_t dy_desc, dx_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    status = cudnnCreateTensorDescriptor(&dy_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroy(cudnn_handle); return; }
    status = cudnnCreateTensorDescriptor(&dx_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroyTensorDescriptor(dy_desc); cudnnDestroy(cudnn_handle); return; }
    status = cudnnCreateFilterDescriptor(&filter_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroyTensorDescriptor(dy_desc); cudnnDestroyTensorDescriptor(dx_desc); cudnnDestroy(cudnn_handle); return; }
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroyTensorDescriptor(dy_desc); cudnnDestroyTensorDescriptor(dx_desc); cudnnDestroyFilterDescriptor(filter_desc); cudnnDestroy(cudnn_handle); return; }

    status = cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        batch, in_channels, length_in, 1);
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;
    status = cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        batch, out_channels, length_out_base, 1);
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;
    status = cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        out_channels, in_channels, kernel_size, 1);
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;
    status = cudnnSetConvolution2dDescriptor(conv_desc,
                                             padding, 0, // pad_h, pad_w
                                             stride, 1,  // stride_h, stride_w
                                             1, 1,       // dilation_h, dilation_w
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;

    // Transpose weight: allocate device memory for transposed weight
    float* weight_T = nullptr;
    if (cudaMalloc(&weight_T, sizeof(float) * out_channels * in_channels * kernel_size) != cudaSuccess) {
        goto cleanup;
    }
    {
        int total_weight = in_channels * out_channels * kernel_size;
        int block_size = 256;
        int grid_size = (total_weight + block_size - 1) / block_size;
        transpose_weight_kernel<<<grid_size, block_size, 0, stream>>>(
            weight, weight_T, in_channels, out_channels, kernel_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(weight_T);
            goto cleanup;
        }
    }

    // Allocate temporary base output buffer on device
    float* base_output = nullptr;
    size_t base_size = (size_t)batch * out_channels * length_out_base;
    if (cudaMalloc(&base_output, sizeof(float) * base_size) != cudaSuccess) {
        cudaFree(weight_T);
        goto cleanup;
    }

    // Get algorithm for backward data
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    status = cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &bwd_data_algo);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(weight_T);
        cudaFree(base_output);
        goto cleanup;
    }

    // Compute workspace size
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        bwd_data_algo,
        &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(weight_T);
        cudaFree(base_output);
        goto cleanup;
    }

    void* workspace = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            cudaFree(weight_T);
            cudaFree(base_output);
            goto cleanup;
        }
    }

    // Run backward data convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_T,
        dy_desc,
        x,
        conv_desc,
        bwd_data_algo,
        workspace,
        workspace_size,
        &beta,
        dx_desc,
        base_output);

    if (workspace) cudaFree(workspace);
    cudaFree(weight_T);

    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(base_output);
        goto cleanup;
    }

    // Add bias if provided (bias pointer non-null)
    if (bias != nullptr) {
        // Create bias tensor descriptor: [1, out_channels, 1, 1]
        cudnnTensorDescriptor_t bias_desc;
        status = cudnnCreateTensorDescriptor(&bias_desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudaFree(base_output);
            goto cleanup;
        }
        status = cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            1, out_channels, 1, 1);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(bias_desc);
            cudaFree(base_output);
            goto cleanup;
        }
        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f; // accumulate into base_output
        status = cudnnAddTensor(
            cudnn_handle,
            &alpha_bias,
            bias_desc,
            bias,
            &beta_bias,
            dx_desc,
            base_output);
        cudnnDestroyTensorDescriptor(bias_desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudaFree(base_output);
            goto cleanup;
        }
    }

    // Now copy base_output (with bias) to final output tensor, handling output_padding.
    // If output_padding == 0, final output size equals base, can copy entire buffer.
    if (output_padding == 0) {
        cudaMemcpyAsync(output, base_output, sizeof(float) * base_size, cudaMemcpyDeviceToDevice, stream);
    } else {
        // Need to copy each (batch, channel) segment of length_out_base into larger output.
        // Final output is assumed contiguous and zero-initialized by caller.
        // We use cudaMemsetAsync on output to ensure zeros, then copy each segment.
        size_t total_out_size = (size_t)batch * out_channels * length_out;
        cudaMemsetAsync(output, 0, sizeof(float) * total_out_size, stream);
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                const float* src = base_output + (b * out_channels + c) * length_out_base;
                float* dst = output + (b * out_channels + c) * length_out;
                cudaMemcpyAsync(dst, src, sizeof(float) * length_out_base, cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    cudaFree(base_output);

cleanup:
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
}
