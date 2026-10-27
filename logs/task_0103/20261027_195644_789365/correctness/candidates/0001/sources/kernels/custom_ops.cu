#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------
// Tile kernel: repeats the batch dimension of a 3D tensor [N, C, L] -> [N*R, C, L]
// Input and output are contiguous float tensors.
// Each thread copies one output element. Grid-stride loop handles large sizes.
// -----------------------------------------------------------------------------
__global__ void tile_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t total_elements,
                            int64_t N,
                            int64_t C,
                            int64_t L,
                            int64_t R) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t out_batch = i / (C * L);          // output batch index
        int64_t in_batch = out_batch % N;         // source batch index in original input
        int64_t pos_in_batch = i - out_batch * (C * L); // position within [C, L]
        int64_t src_idx = in_batch * (C * L) + pos_in_batch;
        output[i] = input[src_idx];
    }
}

// Host launcher for tile_forward
// input: pointer to input tensor (contiguous, shape [N, C, L])
// output: pointer to output tensor (contiguous, shape [N*R, C, L])
// N, C, L: input dimensions
// R: repeats
// stream: CUDA stream
void tile_forward_launcher(const float* input, float* output,
                           int64_t N, int64_t C, int64_t L, int64_t R,
                           cudaStream_t stream) {
    int64_t total = N * R * C * L;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks to avoid grid dimension overflow, grid-stride loop handles rest.
    if (blocks > 65535) blocks = 65535;
    tile_kernel<<<blocks, threads, 0, stream>>>(input, output, total, N, C, L, R);
}

// -----------------------------------------------------------------------------
// Conv1d forward using cuDNN
// This function is a host launcher with no PyTorch dependency.
// It assumes input, weight, bias are all CUDA float32 contiguous tensors.
// Parameters:
//   x: input [N, C_in, L]
//   weight: [C_out, C_in, kernel_size]
//   bias: [C_out] or nullptr if no bias (but contract says bias exists)
//   N, C_in, C_out, kernel_size, L: dimensions
//   stride, pad, dilation: conv parameters
//   groups: group count (1 for standard conv)
//   stream: CUDA stream
// Returns 0 on success, negative on error (check after call).
// -----------------------------------------------------------------------------
int conv1d_forward_launcher(const float* x, const float* weight, const float* bias,
                            float* y,
                            int N, int C_in, int C_out, int kernel_size, int L,
                            int stride, int pad, int dilation, int groups,
                            cudaStream_t stream) {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) return -1;

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set input tensor: N, C_in, L (for conv1d, height=1, width=L)
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, C_in, 1, L);
    
    int out_L = (L + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, C_out, 1, out_L);

    // Filter: C_out, C_in, 1, kernel_size
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               C_out, C_in, 1, kernel_size);

    // Convolution descriptor: 2D but with height=1, effectively 1D conv
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad, 0,        // pad_h, pad_w (pad_h should be padding, pad_w=0? Actually for 1D we pad width)
                                    stride, 1,     // stride_h, stride_w
                                    dilation, 1,   // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    // Note: For conv1d, we set height dimension to 1 and use pad_h for padding.
    // Since kernel height = 1, pad_h should be the actual padding along length dimension.
    // In the descriptor above, pad_h = pad, pad_w = 0 because width dim has no padding? Wait, we treat L as width, height=1.
    // Actually for a 1D convolution represented as 2D with H=1, the convolution operates along width W.
    // The padding should be along W dimension, not H. So we need pad_w = pad, pad_h = 0.
    // Let's fix by setting pad_h=0, pad_w=pad.
    // Re-create conv_desc with correct padding:
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, pad,        // pad_h=0, pad_w=pad
                                    stride, 1,     // stride_h, stride_w (stride along H is irrelevant, set 1)
                                    dilation, 1,   // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0, &algo);

    // Workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc,
                                            algo, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Set algorithm
    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionForward(handle,
                                     &alpha,
                                     x_desc, x,
                                     w_desc, weight,
                                     conv_desc, algo,
                                     workspace, workspace_size,
                                     &beta,
                                     y_desc, y);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(handle);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        if (workspace) cudaFree(workspace);
        return -2;
    }

    // Add bias
    if (bias != nullptr) {
        // Use cudnnAddTensor or a simple kernel; we use cudnnAddTensor for brevity.
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, C_out, 1, 1);
        cudnnAddTensor(handle, &alpha, bias_desc, bias, &alpha, y_desc, y);
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    // Cleanup
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace) cudaFree(workspace);

    return 0;
}

// Convenience wrapper for tile and conv used by binding, but binding will call the individual launchers.
// However, we also provide extern "C" names for compatibility.
extern "C" {
    void tile_launcher(const float* input, float* output, int64_t N, int64_t C, int64_t L, int64_t R, cudaStream_t stream) {
        tile_forward_launcher(input, output, N, C, L, R, stream);
    }
    int conv1d_launcher(const float* x, const float* weight, const float* bias, float* y,
                        int N, int C_in, int C_out, int kernel_size, int L,
                        int stride, int pad, int dilation, int groups, cudaStream_t stream) {
        return conv1d_forward_launcher(x, weight, bias, y, N, C_in, C_out, kernel_size, L,
                                       stride, pad, dilation, groups, stream);
    }
}
