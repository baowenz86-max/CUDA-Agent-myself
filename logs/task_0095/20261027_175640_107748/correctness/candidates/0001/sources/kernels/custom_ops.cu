// kernels/custom_ops.cu
// CUDA kernels and launchers for custom ops (no PyTorch dependencies)
// All functions use raw pointers, scalars, and cudaStream_t

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------
// linspace kernel
// Generates `steps` float32 values from start to end inclusive,
// matching torch.linspace (endpoint included) and then moved to CUDA.
// If steps == 1, output is [start].
// -----------------------------------------------------------------------------
__global__ void linspace_kernel(float* output, float start, float end, int64_t steps) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        if (steps == 1) {
            output[idx] = start;
        } else {
            float ratio = static_cast<float>(idx) / static_cast<float>(steps - 1);
            output[idx] = start + ratio * (end - start);
        }
    }
}

// -----------------------------------------------------------------------------
// sub_broadcast kernel
// Computes out[n, c, l] = x[n, c, l] - linear_space[l]
// x may be non-contiguous: we use element-wise indexing with strides.
// linear_space must be contiguous.
// -----------------------------------------------------------------------------
__global__ void sub_broadcast_kernel(
    float* out,
    const float* x,
    const float* linear_space,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_length
) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t total = batch * channels * length;
    if (idx >= total) return;

    int64_t l = idx % length;
    int64_t rem = idx / length;
    int64_t c = rem % channels;
    int64_t n = rem / channels;

    int64_t x_idx = n * stride_batch + c * stride_channel + l * stride_length;
    out[idx] = x[x_idx] - linear_space[l];
}

// -----------------------------------------------------------------------------
// cuDNN ConvTranspose1d launcher
// Parameters:
//   input:   [batch, in_channels, length_in]  (float32, may be non-contiguous)
//   weight:  [in_channels, out_channels, kernel_size] (contiguous, float32)
//   bias:    [out_channels] or nullptr
//   stride, padding, output_padding
// Returns via output pointer allocated by the caller (but launcher can optionally allocate,
// here caller passes output pointer).
// -----------------------------------------------------------------------------
extern "C" void conv_transpose1d_launcher(
    const float* input,
    int64_t batch,
    int64_t in_channels,
    int64_t length_in,
    const float* weight,
    int64_t out_channels,
    int64_t kernel_size,
    const float* bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    float* output,
    int64_t length_out,
    cudaStream_t stream
) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnSetStream(handle, stream);

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set input descriptor: shape [batch, in_channels, length_in]
    cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        static_cast<int>(in_channels),
        static_cast<int>(length_in),
        1
    );

    // Set filter descriptor: shape [in_channels, out_channels, kernel_size, 1]
    // For ConvTranspose1d, cuDNN expects filter as [in_channels, out_channels, kernel_size, 1]
    cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(kernel_size),
        1
    );

    // Set convolution descriptor: we need cuDNN backward data operation
    // with stride and padding; output_padding is handled by choosing output size
    // and setting proper padding mode.
    cudnnSetConvolution2dDescriptor(
        conv_desc,
        static_cast<int>(padding),      // pad_h (use same for width but we use height as length)
        0,                               // pad_w (not used effectively)
        static_cast<int>(stride),       // stride_h
        1,                               // stride_w
        1,                               // dilation_h
        1,                               // dilation_w
        CUDNN_CROSS_CORRELATION,         // mode (ConvTranspose1d uses cross-correlation in PyTorch)
        CUDNN_DATA_FLOAT
    );

    // Output descriptor: [batch, out_channels, length_out, 1]
    cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        static_cast<int>(out_channels),
        static_cast<int>(length_out),
        1
    );

    // Bias descriptor: [1, out_channels, 1, 1]
    if (bias != nullptr) {
        cudnnSetTensor4dDescriptor(
            bias_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            static_cast<int>(out_channels),
            1,
            1
        );
    }

    // For ConvTranspose, we use cudnnConvolutionBackwardData
    // No need for weight gradients; all pointers are const float* but cuDNN expects void*,
    // so const_cast is used (input is read-only, output is written).
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnStatus_t status = cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc,
        weight,
        input_desc,
        input,
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        nullptr,
        0,
        &beta,
        output_desc,
        output
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cudnnConvolutionBackwardData failed: %s\n", cudnnGetErrorString(status));
    }

    // Add bias if present
    if (bias != nullptr) {
        // use cudnnAddTensor
        cudnnStatus_t bias_status = cudnnAddTensor(
            handle,
            &alpha,
            bias_desc,
            bias,
            &alpha,
            output_desc,
            output
        );
        if (bias_status != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "cudnnAddTensor failed: %s\n", cudnnGetErrorString(bias_status));
        }
    }

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
}

// -----------------------------------------------------------------------------
// Host launchers with plain C interface for binding layer
// -----------------------------------------------------------------------------
extern "C" void linspace_launcher(
    float* output,
    float start,
    float end,
    int64_t steps,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t blocks = (steps + block_size - 1) / block_size;
    linspace_kernel<<<static_cast<int>(blocks), block_size, 0, stream>>>(
        output, start, end, steps
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "linspace_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void sub_broadcast_launcher(
    float* output,
    const float* x,
    const float* linear_space,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_length,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t total = batch * channels * length;
    int64_t blocks = (total + block_size - 1) / block_size;
    sub_broadcast_kernel<<<static_cast<int>(blocks), block_size, 0, stream>>>(
        output, x, linear_space,
        batch, channels, length,
        stride_batch, stride_channel, stride_length
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sub_broadcast_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
