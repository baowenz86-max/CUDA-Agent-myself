// custom_ops_binding.cpp - PyTorch bindings for conv1d and maxpool1d using cuDNN
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

namespace {

// Helper to get current CUDA stream
cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper to check cudnn status
void checkCudnn(cudnnStatus_t status) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status));
}

// Conv1d forward using cuDNN
torch::Tensor conv1d_cudnn_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "bias must be CUDA tensor or undefined");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_out, C_in, kernel)");

    // Make contiguous if necessary
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(1) == C_in, "weight input channels mismatch");
    if (bias_contig.defined()) {
        TORCH_CHECK(bias_contig.size(0) == C_out, "bias size mismatch");
    }

    // Compute output length
    int64_t L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    // cuDNN handles and descriptors
    cudnnHandle_t handle;
    checkCudnn(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc));
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors for NCHW format (but 1D, use N,C,L,1)
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, L_in, 1));
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, L_out, 1));

    // Filter descriptor: [C_out, C_in, kernel_size]
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kernel_size, 1));

    // Convolution descriptor
    checkCudnn(cudnnSetConvolution2dDescriptor(
        conv_desc, padding, 0, stride, 1, dilation, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));

    // Determine algorithm
    cudnnConvolutionFwdAlgo_t algo;
    checkCudnn(cudnnGetConvolutionForwardAlgorithm(
        handle, x_desc, w_desc, conv_desc, y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo
    ));

    // Workspace size
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size
    ));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Set stream
    cudaStream_t stream = getCurrentStream();
    checkCudnn(cudnnSetStream(handle, stream));

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, input_contig.data_ptr<float>(),
        w_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>()
    ));

    // Add bias if present
    if (bias_contig.defined()) {
        // Use cudnnAddTensor
        cudnnTensorDescriptor_t bias_desc;
        checkCudnn(cudnnCreateTensorDescriptor(&bias_desc));
        checkCudnn(cudnnSetTensor4dDescriptor(
            bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1
        ));
        float alpha_bias = 1.0f, beta_bias = 1.0f; // output += bias
        checkCudnn(cudnnAddTensor(
            handle, &alpha_bias, bias_desc, bias_contig.data_ptr<float>(), &beta_bias, y_desc, output.data_ptr<float>()
        ));
        checkCudnn(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    checkCudnn(cudnnDestroyTensorDescriptor(x_desc));
    checkCudnn(cudnnDestroyTensorDescriptor(y_desc));
    checkCudnn(cudnnDestroyFilterDescriptor(w_desc));
    checkCudnn(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCudnn(cudnnDestroy(handle));

    return output;
}

// MaxPool1d forward using cuDNN
torch::Tensor maxpool1d_forward(
    const torch::Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");

    auto input_contig = input.contiguous();
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t L_in = input_contig.size(2);

    // Compute output length (same formula as PyTorch max_pool1d)
    int64_t L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, L_out}, input_contig.options());

    cudnnHandle_t handle;
    checkCudnn(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnPoolingDescriptor_t pool_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnn(cudnnCreatePoolingDescriptor(&pool_desc));

    // Set tensors as 4D NCHW with H=1, W=L
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, L_in, 1));
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, L_out, 1));

    // Pooling descriptor: window (kernel_size, 1), stride (stride, 1), padding (padding, 0)
    checkCudnn(cudnnSetPooling2dDescriptor(
        pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        kernel_size, 1, padding, 0, stride, 1
    ));

    cudaStream_t stream = getCurrentStream();
    checkCudnn(cudnnSetStream(handle, stream));

    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnPoolingForward(
        handle, pool_desc,
        &alpha, x_desc, input_contig.data_ptr<float>(),
        &beta, y_desc, output.data_ptr<float>()
    ));

    checkCudnn(cudnnDestroyTensorDescriptor(x_desc));
    checkCudnn(cudnnDestroyTensorDescriptor(y_desc));
    checkCudnn(cudnnDestroyPoolingDescriptor(pool_desc));
    checkCudnn(cudnnDestroy(handle));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv1d_cudnn_forward", &conv1d_cudnn_forward, "Conv1d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"));
    m.def("maxpool1d_forward", &maxpool1d_forward, "MaxPool1d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("kernel_size"), pybind11::arg("stride"),
          pybind11::arg("padding"), pybind11::arg("dilation"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);