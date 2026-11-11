#include <cuda_runtime.h>
#include <cudnn.h>
#include <cmath>

// Softplus kernel: y = log(1 + exp(x)) with threshold for numerical stability
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

// Helper to check cudnn status
static inline void checkCudnnInternal(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        // We cannot throw exceptions in .cu launcher, but we can set an error flag
        // For simplicity, we rely on cudaGetLastError later, but cudnn errors won't be visible.
        // Better to abort or ignore? Since this is compiled with nvcc, we can use printf and return.
        printf("cuDNN error: %s, status: %d\n", msg, (int)status);
    }
}

// cuDNN convolution launcher (3D)
extern "C" void conv3d_launcher(
    const float* input,      // input data pointer (contiguous)
    const float* weight,     // weight data pointer (contiguous)
    const float* bias,       // optional bias pointer (can be nullptr)
    float* output,           // output data pointer
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t K, int64_t kD, int64_t kH, int64_t kW,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dilationD, int64_t dilationH, int64_t dilationW,
    cudaStream_t stream
) {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreate failed\n");
        return;
    }
    cudnnSetStream(handle, stream);

    // Create descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors (NCDHW)
    int dimA[5] = {(int)N, (int)C, (int)D, (int)H, (int)W};
    int strideA[5] = {(int)(C*D*H*W), (int)(D*H*W), (int)(H*W), (int)W, 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, dimA, strideA);

    int64_t D_out = (D + 2*padD - dilationD*(kD-1) - 1) / strideD + 1;
    int64_t H_out = (H + 2*padH - dilationH*(kH-1) - 1) / strideH + 1;
    int64_t W_out = (W + 2*padW - dilationW*(kW-1) - 1) / strideW + 1;

    int dimB[5] = {(int)N, (int)K, (int)D_out, (int)H_out, (int)W_out};
    int strideB[5] = {(int)(K*D_out*H_out*W_out), (int)(D_out*H_out*W_out), (int)(H_out*W_out), (int)W_out, 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, dimB, strideB);

    int filterDimA[5] = {(int)K, (int)C, (int)kD, (int)kH, (int)kW};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDimA);

    int padA[3] = {(int)padD, (int)padH, (int)padW};
    int filterStrideA[3] = {(int)strideD, (int)strideH, (int)strideW};
    int dilationA[3] = {(int)dilationD, (int)dilationH, (int)dilationW};
    cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, filterStrideA, dilationA,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Find algorithm and workspace
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc,
                                            algo, &workspace_size);

    // Allocate temporary workspace on device using cudaMalloc
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha,
                            x_desc, input,
                            w_desc, weight,
                            conv_desc, algo,
                            workspace, workspace_size,
                            &beta,
                            y_desc, output);

    // Add bias if present
    if (bias != nullptr) {
        cudnnCreateTensorDescriptor(&b_desc);
        int biasDim[5] = {1, (int)K, 1, 1, 1};
        int biasStride[5] = {(int)K, 1, 1, 1, 1};
        cudnnSetTensorNdDescriptor(b_desc, CUDNN_DATA_FLOAT, 5, biasDim, biasStride);
        float alpha_bias = 1.0f;
        cudnnAddTensor(handle, &alpha_bias, b_desc, bias,
                       &alpha, y_desc, output);
        cudnnDestroyTensorDescriptor(b_desc);
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
}
