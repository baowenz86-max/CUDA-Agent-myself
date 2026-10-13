#include <cuda_runtime.h>
#include <cfloat>
#include <cudnn.h>

// ReLU kernel: out = max(in, 0)
__global__ void relu_forward_kernel(float* __restrict__ out,
                                   const float* __restrict__ in,
                                   int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        out[i] = fmaxf(in[i], 0.0f);
    }
}

// MaxPool2d kernel: NCHW layout, kernel_size=3, stride=2, padding=1, dilation=1
// Output: [N, C, H_out, W_out], H_out = floor((H + 2*padding - kernel_size)/stride + 1)
// Each thread computes one output element.
__global__ void maxpool2d_forward_kernel(float* __restrict__ out,
                                         const float* __restrict__ in,
                                         int N, int C, int H, int W,
                                         int H_out, int W_out,
                                         int kernel_size, int stride,
                                         int padding, int dilation) {
    int total_out = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // Decompose linear index to (n, c, ho, wo)
    int wo = idx % W_out;
    int tmp = idx / W_out;
    int ho = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Input base pointer for this (n,c)
    const float* in_nc = in + ((long long)n * C + c) * H * W;

    float max_val = -FLT_MAX;

    // Iterate over kernel window; apply padding implicitly (skip out-of-bounds)
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = ho * stride - padding + kh * dilation;
        if (h_in < 0 || h_in >= H) continue;
        int base_h_offset = h_in * W;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = wo * stride - padding + kw * dilation;
            if (w_in < 0 || w_in >= W) continue;
            float val = in_nc[base_h_offset + w_in];
            if (val > max_val) max_val = val;
        }
    }

    // Output linear index (same decomposition, as output is contiguous)
    out[idx] = max_val;
}

// Launcher for relu: assumes input and output are contiguous (but supports strided via linear indexing)
// The binding layer ensures both tensors are allocated with appropriate shapes and passed as raw pointers.
extern "C" void relu_forward_launcher(float* out,
                                      const float* in,
                                      long long total_elements,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // cap blocks; grid-stride loop handles rest
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(out, in, (int)total_elements);
}

// Launcher for maxpool2d: assumes input and output are contiguous.
// Parameters are passed from binding layer which checks shapes.
extern "C" void maxpool2d_forward_launcher(float* out,
                                           const float* in,
                                           int N, int C, int H, int W,
                                           int H_out, int W_out,
                                           int kernel_size, int stride,
                                           int padding, int dilation,
                                           cudaStream_t stream) {
    int total_out = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid-stride loop not needed if capped, but ensure at least one block
    maxpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        out, in, N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
}

// cuDNN conv2d forward launcher: performs convolution + optional bias
// All tensors are expected to be contiguous (input may be non-contiguous, but we assume binding passes contiguous)
// Input, weight, bias, output are raw device pointers.
// Dimensions: N, C, H, W for input; K, R, S for weight; output dimensions computed inside.
extern "C" void conv2d_forward_launcher(const float* input, const float* weight, const float* bias,
                                        float* output,
                                        int N, int C, int H, int W,
                                        int K, int R, int S,
                                        int stride_h, int stride_w,
                                        int pad_h, int pad_w,
                                        int dilation_h, int dilation_w,
                                        int has_bias,
                                        cudaStream_t stream) {
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, stream);

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Compute output dimensions
    int H_out = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, C, H, W);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, K, H_out, W_out);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, K, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               K, C, R, S);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                            input_desc,
                                            filter_desc,
                                            conv_desc,
                                            output_desc,
                                            algo,
                                            &workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle,
                            &alpha,
                            input_desc, input,
                            filter_desc, weight,
                            conv_desc,
                            algo,
                            workspace, workspace_size,
                            &beta,
                            output_desc, output);

    if (has_bias) {
        beta = 1.0f;
        cudnnAddTensor(cudnn_handle,
                       &alpha,
                       bias_desc, bias,
                       &beta,
                       output_desc, output);
    }

    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
}
