#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------
// custom_silu: elementwise SiLU (x * sigmoid(x)) for float32
// -----------------------------------------------------------------------------
__global__ void silu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        float x = input[i];
        // Avoid overflow in exp for large positive x; clamp to [-88, 88]
        float exp_val = __expf(-__saturatef(__fabsf(x)));
        // sigmoid(x) = 1 / (1 + exp(-x))
        // For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
        // For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
        float sig;
        if (x >= 0.0f) {
            sig = 1.0f / (1.0f + __expf(-x));
        } else {
            float ex = __expf(x);
            sig = ex / (1.0f + ex);
        }
        output[i] = x * sig;
    }
}

extern "C" cudaError_t custom_silu_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    // Cap blocks to reasonable max grid (2^31-1) but grid-stride loop handles large sizes
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL;
    int blocks = static_cast<int>(blocks64);
    silu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
    return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// custom_conv_transpose1d using cuDNN
// -----------------------------------------------------------------------------
// We assume binding.cpp will:
//   1. copy input/weight/bias to contiguous CUDA float32 tensors if needed
//   2. allocate output tensor of shape [N, C_out, L_out]
//   3. create and configure cudnnHandle_t
//   4. create cudnnTensorDescriptor_t for input/output
//   5. create cudnnFilterDescriptor_t for weight
//   6. call this launcher with raw pointers and descriptors, stream
//   7. destroy descriptors afterwards
// To keep this .cu self-contained, we provide the launcher taking all
// cuDNN handles/descriptors and using cudnnConvolutionBackwardData or
// cudnnConvolutionTranspose depending on version.
// We use cudnnConvolutionBackwardData (the canonical way for transposed conv)
// with cudnnConvolutionBwdDataAlgo_t selected by cudnnGetConvolutionBackwardDataAlgorithm_v7.
// Actually, cudnn has cudnnConvolutionBackwardData but that requires a descriptor
// for the forward conv. Alternatively, we can use cudnnConvolution with
// CUDNN_CROSS_CORRELATION and swapped strides? Simpler: use cudnnConvolutionBackwardData
// because PyTorch ConvTranspose1d uses that under the hood.

extern "C" cudnnStatus_t custom_conv_transpose1d_cudnn_launcher(
    cudnnHandle_t handle,
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t L_in,
    int64_t L_out,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream
) {
    cudnnStatus_t status;

    // Set the stream
    status = cudnnSetStream(handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) return status;

    // Create tensor descriptors for input and output (NCHW format, but for 1D we use 2D with height=1)
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnFilterDescriptor_t w_desc;

    status = cudnnCreateTensorDescriptor(&x_desc);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnCreateTensorDescriptor(&y_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroyTensorDescriptor(x_desc); return status; }
    status = cudnnCreateFilterDescriptor(&w_desc);
    if (status != CUDNN_STATUS_SUCCESS) { cudnnDestroyTensorDescriptor(x_desc); cudnnDestroyTensorDescriptor(y_desc); return status; }

    // For 1D conv, cuDNN expects 3D tensors: N, C, 1, L
    // Input: [N, C_in, 1, L_in]
    int dimA[4] = { (int)N, (int)C_in, 1, (int)L_in };
    int strideA[4] = { (int)(C_in * 1 * L_in), (int)(1 * L_in), (int)L_in, 1 };
    status = cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        dimA[0], dimA[1], dimA[2], dimA[3]
    );
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;

    // Output: [N, C_out, 1, L_out]
    int dimB[4] = { (int)N, (int)C_out, 1, (int)L_out };
    status = cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        dimB[0], dimB[1], dimB[2], dimB[3]
    );
    if (status != CUDNN_STATUS_SUCCESS) goto cleanup;

    // Weight: [C_in, C_out, kernel_size] -> cuDNN expects [K, C, 1, R] where K = C_out, C = C_in
    // But the weight tensor from PyTorch shape is [C_in, C_out, kernel_size].
    // cuDNN filter descriptor format: [K, C, H, W], where K is output channels, C is input channels.
    // For 1D conv flatten, we map: K = C_out, C = C_in, H = 1, W = kernel_size.
    // However, weight memory layout is [C_in, C_out, kernel]
    // cuDNN expects contiguous KCRS layout: K (out), C (in), R (kernel)
    // Therefore, we need to permute weight dimensions? Actually, cuDNN expects filter in KCRS,
    // while PyTorch weight is [in_channels, out_channels, kernel_size], i.e., C K R.
    // So we must transpose the weight before passing to cuDNN.
    // We'll create a temporary contiguous buffer of shape [C_out, C_in, kernel] and copy/transpose.
    // Alternatively, we can set the filter descriptor with strides to interpret the memory as transposed.
    // But cudnnSetFilter4dDescriptor only accepts contiguous layout? It supports strides.
    // We can use cudnnSetFilterNdDescriptor with custom strides.
    // Simpler: allocate temp and transpose on device using a small kernel or cudaMemcpy2D?
    // For simplicity, we'll create a transposed copy on host? Better on device.
    // We'll use a small kernel to transpose weight: output[out][in][k] = input[in][out][k]
    // Since kernel_size likely 3, C_in=8, C_out=16, very small. We'll do a simple kernel.
    // We'll include a weight transpose kernel.
}
