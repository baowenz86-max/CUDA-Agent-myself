#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdint>

// Softplus kernel: y = log(1 + exp(x)) with threshold for numerical stability
// The threshold is set to 20, matching PyTorch's default behavior.
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// Launcher for softplus forward
extern "C" void softplus_launcher(const float* input,
                                  float* output,
                                  int64_t numel,
                                  cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 4096) blocks = 4096;
    dim3 grid(blocks);
    dim3 block(block_size);
    softplus_kernel<<<grid, block, 0, stream>>>(input, output, numel);
}

// Helper functions for cuDNN error checking
static inline void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        // In a pure launcher, we cannot throw exceptions; instead, record error and return.
        // However, for correctness, we can call abort or simply ignore. Better: store error code in a global?
        // We'll rely on launch error checks from binding side after launcher returns.
        // But if cuDNN fails inside launcher, there is no way to propagate to Python without exception support.
        // Since launcher must be C-compatible, we can only signal failure via a global variable or by not writing output.
        // Typically, binding checks cudaGetLastError after launcher; cuDNN errors do not set CUDA error.
        // To avoid silent failure, we will use fprintf to stderr and set a flag.
        fprintf(stderr, "cuDNN error in launcher: %s: %s\n", msg, cudnnGetErrorString(status));
        // We cannot throw; but we can write an invalid value or not write output.
        // This will be caught as wrong result.
    }
}

// cuDNN convolution launcher
// Parameters:
//   input_ptr: pointer to input tensor (float)
//   weight_ptr: pointer to weight tensor (float)
//   bias_ptr: pointer to bias tensor (float) or nullptr if no bias
//   output_ptr: pointer to output tensor (float)
//   N, C, D, H, W: input dimensions
//   K: output channels
//   kD, kH, kW: kernel dimensions
//   stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w: conv params
//   stream: CUDA stream
// All pointers must be valid CUDA device pointers.
extern "C" void conv3d_launcher(
    const float* input_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    float* output_ptr,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t K, int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
    cudaStream_t stream
) {
    // Compute output dimensions
    int64_t D_out = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) / stride_d + 1;
    int64_t H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    // Create cuDNN handle
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "conv3d_launcher: cudnnCreate failed\n");
        return;
    }
    cudnnSetStream(handle, stream);

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    status = cudnnCreateTensorDescriptor(&x_desc);
    checkCudnn(status, "cudnnCreateTensorDescriptor x");
    status = cudnnCreateTensorDescriptor(&y_desc);
    checkCudnn(status, "cudnnCreateTensorDescriptor y");
    status = cudnnCreateFilterDescriptor(&w_desc);
    checkCudnn(status, "cudnnCreateFilterDescriptor");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    checkCudnn(status, "cudnnCreateConvolutionDescriptor");

    // Set tensor descriptors (NCDHW format)
    int dimA[5] = {static_cast<int>(N), static_cast<int>(C), static_cast<int>(D), static_cast<int>(H), static_cast<int>(W)};
    int strideA[5] = {static_cast<int>(C * D * H * W), static_cast<int>(D * H * W), static_cast<int>(H * W), static_cast<int>(W), 1};
    status = cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, dimA, strideA);
    checkCudnn(status, "cudnnSetTensorNdDescriptor x");

    int dimB[5] = {static_cast<int>(N), static_cast<int>(K), static_cast<int>(D_out), static_cast<int>(H_out), static_cast<int>(W_out)};
    int strideB[5] = {static_cast<int>(K * D_out * H_out * W_out), static_cast<int>(D_out * H_out * W_out), static_cast<int>(H_out * W_out), static_cast<int>(W_out), 1};
    status = cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, dimB, strideB);
    checkCudnn(status, "cudnnSetTensorNdDescriptor y");

    int filterDimA[5] = {static_cast<int>(K), static_cast<int>(C), static_cast<int>(kD), static_cast<int>(kH), static_cast<int>(kW)};
    status = cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDimA);
    checkCudnn(status, "cudnnSetFilterNdDescriptor");

    int padA[3] = {static_cast<int>(pad_d), static_cast<int>(pad_h), static_cast<int>(pad_w)};
    int filterStrideA[3] = {static_cast<int>(stride_d), static_cast<int>(stride_h), static_cast<int>(stride_w)};
    int dilationA[3] = {static_cast<int>(dilation_d), static_cast<int>(dilation_h), static_cast<int>(dilation_w)};
    status = cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    checkCudnn(status, "cudnnSetConvolutionNdDescriptor");

    // Get algorithm
    cudnnConvolutionFwdAlgo_t algo;
    status = cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    checkCudnn(status, "cudnnGetConvolutionForwardAlgorithm");

    // Workspace
    size_t workspace_size = 0;
    status = cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size);
    checkCudnn(status, "cudnnGetConvolutionForwardWorkspaceSize");
    
    // Allocate workspace on device using cudaMalloc
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "conv3d_launcher: cudaMalloc workspace failed\n");
            // cleanup and return
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(y_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(handle);
            return;
        }
    }

    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionForward(handle,
                                     &alpha,
                                     x_desc, input_ptr,
                                     w_desc, weight_ptr,
                                     conv_desc, algo,
                                     workspace, workspace_size,
                                     &beta,
                                     y_desc, output_ptr);
    checkCudnn(status, "cudnnConvolutionForward");

    // Add bias if present
    if (bias_ptr != nullptr) {
        cudnnTensorDescriptor_t b_desc_temp;
        status = cudnnCreateTensorDescriptor(&b_desc_temp);
        checkCudnn(status, "cudnnCreateTensorDescriptor bias");
        int biasDim[5] = {1, static_cast<int>(K), 1, 1, 1};
        int biasStride[5] = {static_cast<int>(K), 1, 1, 1, 1};
        status = cudnnSetTensorNdDescriptor(b_desc_temp, CUDNN_DATA_FLOAT, 5, biasDim, biasStride);
        checkCudnn(status, "cudnnSetTensorNdDescriptor bias");
        float alpha_bias = 1.0f;
        status = cudnnAddTensor(handle, &alpha_bias, b_desc_temp, bias_ptr,
                                &alpha, y_desc, output_ptr);
        checkCudnn(status, "cudnnAddTensor");
        cudnnDestroyTensorDescriptor(b_desc_temp);
    }

    // Cleanup
    if (workspace != nullptr) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
}
