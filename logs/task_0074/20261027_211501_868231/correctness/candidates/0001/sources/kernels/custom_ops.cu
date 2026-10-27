// custom_ops.cu - all CUDA kernels and launchers for this model
#include <cuda_runtime.h>
#include <cstdint>

// SiLU kernel: y = x * sigmoid(x)
__global__ void silu_kernel(float* __restrict__ out, const float* __restrict__ in, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = in[i];
        float sig = 1.0f / (1.0f + expf(-x));
        out[i] = x * sig;
    }
}

extern "C" void silu_forward_launcher(
    float* out,
    const float* in,
    int64_t n,
    cudaStream_t stream) {

    if (n <= 0) return;

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    // Cap blocks to avoid excessive launch config
    if (blocks > 65535) blocks = 65535;

    silu_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// cuDNN-based transposed convolution. 1D transposed conv is implemented via cuDNN as a 2D convolution.
// Input tensor shape (N, C_in, L_in) is treated as (N, C_in, 1, L_in).
// Output tensor shape (N, C_out, L_out) is treated as (N, C_out, 1, L_out).
// Weight shape (C_in, C_out, K) is treated as (C_in, C_out, 1, K).
// cuDNN uses NCHW memory layout with height=1.
#include <cudnn.h>

extern "C" void conv_transpose1d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t length_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    cudaStream_t stream) {

    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 || length_in <= 0 || kernel_size <= 0) return;

    // Output length formula: (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    cudnnSetStream(handle, stream);

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors with NCHW layout, height=1
    cudnnSetTensor4dDescriptor(x_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               (int)batch,
                               (int)in_channels,
                               1,
                               (int)length_in);
    cudnnSetTensor4dDescriptor(y_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               (int)batch,
                               (int)out_channels,
                               1,
                               (int)length_out);
    // Filter: (C_in, C_out, 1, K)
    int filter_dim[4] = {(int)in_channels, (int)out_channels, 1, (int)kernel_size};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dim);

    // Convolution descriptor: transposed conv, zero padding in height and width?
    // For 1D transposed conv with stride, padding, output_padding, we set:
    // padA[0]=0, padA[1]=padding, strideA[0]=1, strideA[1]=stride, dilationA=[1,1],
    // and use CUDNN_CONVOLUTION mode with CUDNN_CROSS_CORRELATION (PyTorch uses cross-correlation)
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, // pad_h (height)
                                    (int)padding, // pad_w
                                    1, // stride_h
                                    (int)stride, // stride_w
                                    1, 1, // dilation
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Algorithm selection
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                           x_desc,
                                           w_desc,
                                           conv_desc,
                                           y_desc,
                                           1,
                                           0,
                                           &algo);

    // Workspace
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            x_desc,
                                            w_desc,
                                            conv_desc,
                                            y_desc,
                                            algo,
                                            &workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle,
                            &alpha,
                            x_desc, in,
                            w_desc, weight,
                            conv_desc, algo,
                            workspace, workspace_size,
                            &beta,
                            y_desc, out);

    if (workspace_size > 0) {
        cudaFree(workspace);
    }

    // Add bias if present
    if (bias != nullptr) {
        // bias shape [out_channels], need to add to each batch and each length position
        // Use a simple kernel: for each (b, oc, 1, l) add bias[oc]
        int64_t total_elements = batch * out_channels * length_out;
        dim3 block(256);
        dim3 grid((unsigned int)((total_elements + block.x - 1) / block.x));
        // limit grid
        if (grid.x > 65535) grid.x = 65535;
        bias_add_kernel<<<grid, block, 0, stream>>>(out, bias, batch, out_channels, length_out);
    }

    // cleanup descriptors
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    // Note: We deliberately do not destroy handle to keep it for subsequent calls; it's static.
}

__global__ void bias_add_kernel(float* __restrict__ out, const float* __restrict__ bias,
                                int64_t batch, int64_t out_channels, int64_t length_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = batch * out_channels * length_out;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t oc = (i / length_out) % out_channels;
        out[i] += bias[oc];
    }
}
