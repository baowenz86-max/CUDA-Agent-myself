#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(float* out, const float* in, long long total_elements, cudaStream_t stream);
extern "C" void maxpool2d_forward_launcher(float* out, const float* in,
                                          int N, int C, int H, int W,
                                          int H_out, int W_out,
                                          int kernel_size, int stride,
                                          int padding, int dilation,
                                          cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while(0)

// Helper to check cuDNN errors
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status)); \
} while(0)

// Conv2d forward using cuDNN
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                             int64_t stride, int64_t padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (out_channels, in_channels, kh, kw)");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input channels must match weight in_channels");

    // Bias is optional; if provided, must be CUDA float32 with correct size
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias.size(0) == weight.size(0), "Bias size must match out_channels");
    }

    // cuDNN requires weight in NCHW format (out_channels, in_channels, kh, kw)
    // Weight already in that format from PyTorch conv2d.
    // Ensure weight and bias are contiguous
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.defined()) {
        bias_contig = bias.contiguous();
    }

    // Get dimensions
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t K = weight_contig.size(0); // out_channels
    int64_t R = weight_contig.size(2); // kernel height
    int64_t S = weight_contig.size(3); // kernel width
    int64_t pad_h = padding;
    int64_t pad_w = padding;
    int64_t stride_h = stride;
    int64_t stride_w = stride;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;

    // Compute output dimensions
    int64_t H_out = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    // Allocate output tensor
    auto output = torch::empty({N, K, H_out, W_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Describe tensors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (NCHW layout)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)N, (int)C, (int)H, (int)W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)N, (int)K, (int)H_out, (int)W_out));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, (int)K, 1, 1));

    // Set filter descriptor: format NCHW (K, C, R, S)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           (int)K, (int)C, (int)R, (int)S));

    // Set convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                (int)pad_h, (int)pad_w,
                                                (int)stride_h, (int)stride_w,
                                                (int)dilation_h, (int)dilation_w,
                                                CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
                                                CUDNN_DATA_FLOAT));

    // Find best algorithm (optional; can use CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                        input_desc,
                                                        filter_desc,
                                                        conv_desc,
                                                        output_desc,
                                                        algo,
                                                        &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Perform convolution
    float alpha = 1.0f;
    float beta = 0.0f; // output initially zero, so beta=0
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle,
                                        &alpha,
                                        input_desc, input.data_ptr<float>(),
                                        filter_desc, weight_contig.data_ptr<float>(),
                                        conv_desc,
                                        algo,
                                        workspace, workspace_size,
                                        &beta,
                                        output_desc, output.data_ptr<float>()));

    // Add bias if provided
    if (bias.defined()) {
        beta = 1.0f; // accumulate on output
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                                   &alpha,
                                   bias_desc, bias_contig.data_ptr<float>(),
                                   &beta,
                                   output_desc, output.data_ptr<float>()));
    }

    // Cleanup
    if (workspace) {
        CHECK_CUDA(cudaFree(workspace));
    }
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    return output;
}

// ReLU forward wrapper
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous for relu_forward");

    auto output = torch::empty_like(input);
    long long total_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(), total_elements, stream);
    CHECK_CUDA(cudaGetLastError());

    return output;
}

// MaxPool2d forward wrapper
torch::Tensor maxpool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride,
                               int64_t padding, int64_t dilation, bool ceil_mode) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous for maxpool2d_forward");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Compute output dimensions (PyTorch formula)
    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (ceil_mode) {
        // PyTorch uses ceil for output size if ceil_mode=True, but here we default false
        // But we still implement the correct formula if needed
        H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
        W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    }

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool2d_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                               (int)N, (int)C, (int)H, (int)W,
                               (int)H_out, (int)W_out,
                               (int)kernel_size, (int)stride,
                               (int)padding, (int)dilation,
                               stream);
    CHECK_CUDA(cudaGetLastError());

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward (cuDNN)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("input"));
    m.def("maxpool2d_forward", &maxpool2d_forward, "MaxPool2d forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
