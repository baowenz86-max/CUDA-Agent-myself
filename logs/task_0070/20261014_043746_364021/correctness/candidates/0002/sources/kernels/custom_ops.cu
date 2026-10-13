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

// Launcher for relu
 extern "C" void relu_forward_launcher(float* out,
                                       const float* in,
                                       long long total_elements,
                                       cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(out, in, (int)total_elements);
}

// Launcher for maxpool2d
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
    if (blocks > 65535) blocks = 65535;
    maxpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        out, in, N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
}

// Helper to check cuDNN errors inside .cu (only used in this file)
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        return; \
    } \
} while(0)

// cuDNN convolution launcher: performs conv2d with bias, returns raw pointer (output allocated in binding)
extern "C" void conv2d_forward_launcher(
    const float* input,   // input data pointer (NCHW, may be non-contiguous but cuDNN handles strides) - but we'll assume contiguous after binding makes it contiguous
    const float* weight,  // weight data pointer (contiguous)
    const float* bias,    // bias data pointer (contiguous, may be nullptr)
    float* output,        // output data pointer (contiguous)
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    // Compute output dims
    int H_out = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    if (cudnnCreate(&cudnn_handle) != CUDNN_STATUS_SUCCESS) return;
    if (cudnnSetStream(cudnn_handle, stream) != CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    if (cudnnCreateTensorDescriptor(&input_desc) != CUDNN_STATUS_SUCCESS) goto cleanup_handle;
    if (cudnnCreateTensorDescriptor(&output_desc) != CUDNN_STATUS_SUCCESS) goto cleanup_input;
    if (cudnnCreateTensorDescriptor(&bias_desc) != CUDNN_STATUS_SUCCESS) goto cleanup_output;
    if (cudnnCreateFilterDescriptor(&filter_desc) != CUDNN_STATUS_SUCCESS) goto cleanup_bias;
    if (cudnnCreateConvolutionDescriptor(&conv_desc) != CUDNN_STATUS_SUCCESS) goto cleanup_filter;

    // Set tensor descriptors (assuming contiguous input; binding will make input contiguous)
    if (cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   N, C, H, W) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;
    if (cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   N, K, H_out, W_out) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;
    if (cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, K, 1, 1) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;

    // Filter descriptor
    if (cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   K, C, R, S) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;

    // Convolution descriptor
    if (cudnnSetConvolution2dDescriptor(conv_desc,
                                        pad_h, pad_w,
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;

    // Choose algorithm (use a simple default)
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Get workspace size
    size_t workspace_size = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                input_desc,
                                                filter_desc,
                                                conv_desc,
                                                output_desc,
                                                algo,
                                                &workspace_size) != CUDNN_STATUS_SUCCESS) goto cleanup_conv;

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) goto cleanup_conv;
    }

    // Perform convolution
    float alpha = 1.0f;
    float beta = 0.0f;
    if (cudnnConvolutionForward(cudnn_handle,
                                &alpha,
                                input_desc, input,
                                filter_desc, weight,
                                conv_desc,
                                algo,
                                workspace, workspace_size,
                                &beta,
                                output_desc, output) != CUDNN_STATUS_SUCCESS) {
        if (workspace) cudaFree(workspace);
        goto cleanup_conv;
    }

    // Add bias if provided
    if (bias != nullptr) {
        beta = 1.0f; // accumulate on output
        if (cudnnAddTensor(cudnn_handle,
                           &alpha,
                           bias_desc, bias,
                           &beta,
                           output_desc, output) != CUDNN_STATUS_SUCCESS) {
            if (workspace) cudaFree(workspace);
            goto cleanup_conv;
        }
    }

    // Cleanup
    if (workspace) cudaFree(workspace);

cleanup_conv:
    cudnnDestroyConvolutionDescriptor(conv_desc);
cleanup_filter:
    cudnnDestroyFilterDescriptor(filter_desc);
cleanup_bias:
    cudnnDestroyTensorDescriptor(bias_desc);
cleanup_output:
    cudnnDestroyTensorDescriptor(output_desc);
cleanup_input:
    cudnnDestroyTensorDescriptor(input_desc);
cleanup_handle:
    cudnnDestroy(cudnn_handle);
}
