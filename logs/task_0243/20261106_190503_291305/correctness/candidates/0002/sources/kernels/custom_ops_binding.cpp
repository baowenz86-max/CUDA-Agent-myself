#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare the average pooling launcher from custom_ops.cu
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    cudaStream_t stream);

// Helper to check CUDA and cuDNN errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status)); \
} while (0)

// Average pooling forward wrapper
// kernel_size parameter is actually a tuple (2,2,2) from model_new.py
// We accept std::vector<int64_t> to handle tuple conversion via pybind11/stl.h
torch::Tensor avg_pool3d_forward(torch::Tensor x, std::vector<int64_t> kernel_size) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(kernel_size.size() == 3, "kernel_size must have 3 elements");
    TORCH_CHECK(kernel_size[0] == 2 && kernel_size[1] == 2 && kernel_size[2] == 2, "Only kernel_size=(2,2,2) is supported");

    // For this specific use case: stride=(2,2,2), padding=0
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int D_out = D / 2;
    int H_out = H / 2;
    int W_out = W / 2;

    // Create output tensor
    auto output = torch::empty({N, C, D_out, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        stream
    );

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    return output;
}

// Conv3d forward wrapper using cuDNN
// stride is int, padding is tuple (1,2,2)
torch::Tensor conv3d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    std::vector<int64_t> padding)
{
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "Bias must be CUDA tensor if defined");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D (out_channels, in_channels/groups, kD, kH, kW)");
    TORCH_CHECK(padding.size() == 3, "padding must have 3 elements");

    // Dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int K_out = weight.size(0);
    int C_in_per_group = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    TORCH_CHECK(C_in == C_in_per_group, "For groups=1, weight in_channels must match input channels");

    // Output dimensions with padding=(pd,ph,pw) and stride=(s,s,s)
    int D_out = (D + 2 * padding[0] - kD) / stride + 1;
    int H_out = (H + 2 * padding[1] - kH) / stride + 1;
    int W_out = (W + 2 * padding[2] - kW) / stride + 1;

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));

    // Set stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (NCDHW format)
    int dimA[5] = {N, C_in, D, H, W};
    int strideA[5] = {C_in*D*H*W, D*H*W, H*W, W, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, dimA, strideA));

    int dimB[5] = {N, K_out, D_out, H_out, W_out};
    int strideB[5] = {K_out*D_out*H_out*W_out, D_out*H_out*W_out, H_out*W_out, W_out, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, dimB, strideB));

    int filterDim[5] = {K_out, C_in_per_group, kD, kH, kW};
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDim));

    // Convolution descriptor: stride, padding, dilation
    int padA[3] = {(int)padding[0], (int)padding[1], (int)padding[2]};
    int strideA2[3] = {(int)stride, (int)stride, (int)stride};
    int dilationA[3] = {1, 1, 1};
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, strideA2, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Find algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, x_desc, w_desc, conv_desc, y_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    // Workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Allocate output tensor
    auto output = torch::empty({N, K_out, D_out, H_out, W_out}, x.options());

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        x_desc, x.data_ptr<float>(),
        w_desc, weight.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>()
    ));

    // Add bias if defined
    if (bias.defined()) {
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        int biasDim[5] = {1, K_out, 1, 1, 1};
        int biasStride[5] = {K_out, 1, 1, 1, 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 5, biasDim, biasStride));
        float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias.data_ptr<float>(), &beta_bias, y_desc, output.data_ptr<float>()));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    // Check for CUDA errors after kernel launches
    CHECK_CUDA(cudaGetLastError());

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "3D average pooling forward",
          py::arg("x"), py::arg("kernel_size"));
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward with cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
