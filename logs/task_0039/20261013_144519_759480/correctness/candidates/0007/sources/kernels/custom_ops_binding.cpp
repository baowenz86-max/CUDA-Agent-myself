// custom_ops_binding.cpp
// PyTorch bindings for custom CUDA kernels and cuDNN convolution
// Convolution uses cuDNN; other ops use custom CUDA kernels from custom_ops.cu

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>
#include <iostream>

// Declare launchers from custom_ops.cu
extern "C" void add_bias_launcher(float* out, const float* x, const float* bias,
                                  int N, int C, int H, int W, cudaStream_t stream);

extern "C" void batch_norm_launcher(float* out, const float* x,
                                    const float* weight, const float* bias,
                                    const float* running_mean, const float* running_var,
                                    float eps, int N, int C, int H, int W,
                                    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(float* out, const float* x,
                                    int N, int C, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream);

extern "C" void sigmoid_launcher(float* out, const float* x, int64_t total,
                                 cudaStream_t stream);

// Helper to check CUDA and cuDNN status
static void check_cuda(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}
static void check_cudnn(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// Conv2d using cuDNN
static torch::Tensor conv2d_forward(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be CUDA tensor");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Shapes
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels, "weight input channels mismatch");

    // Parameters: stride=1, padding=0, dilation=1, groups=1
    int stride_h = 1, stride_w = 1;
    int padding_h = 0, padding_w = 0;
    int dilation_h = 1, dilation_w = 1;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Allocate output
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle
    cudnnHandle_t cudnn_handle;
    check_cudnn(cudnnCreate(&cudnn_handle), "cudnnCreate failed");
    check_cudnn(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream failed");

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    check_cudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x failed");
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y failed");
    check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
    check_cudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor failed");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors (NCHW format)
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch, in_channels, in_h, in_w), "cudnnSetTensor4dDescriptor x failed");
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch, out_channels, out_h, out_w), "cudnnSetTensor4dDescriptor y failed");
    check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          1, out_channels, 1, 1), "cudnnSetTensor4dDescriptor bias failed");

    // Filter descriptor
    check_cudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_h, kernel_w), "cudnnSetFilter4dDescriptor failed");

    // Convolution descriptor
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding_h, padding_w,
                                                stride_h, stride_w,
                                                dilation_h, dilation_w,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor failed");

    // Choose algorithm
    cudnnConvolutionFwdAlgo_t algo;
    check_cudnn(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo),
                "cudnnGetConvolutionForwardAlgorithm failed");

    // Workspace
    size_t workspace_size = 0;
    check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
                                                        algo, &workspace_size), "cudnnGetConvolutionForwardWorkspaceSize failed");
    void* workspace = nullptr;
    if (workspace_size > 0) {
        check_cuda(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace failed");
    }

    float alpha = 1.0f, beta = 0.0f;
    if (bias_opt.has_value()) {
        // Use bias add mode: cudnnConvolutionBiasActivationForward
        cudnnActivationDescriptor_t activation_desc;
        check_cudnn(cudnnCreateActivationDescriptor(&activation_desc), "cudnnCreateActivationDescriptor failed");
        check_cudnn(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_IDENTITY,
                                                CUDNN_PROPAGATE_NAN, 0.0), "cudnnSetActivationDescriptor failed");
        beta = 0.0f; // output initially zero, bias added by this function
        check_cudnn(cudnnConvolutionBiasActivationForward(cudnn_handle,
                                                         &alpha,
                                                         x_desc, x.data_ptr<float>(),
                                                         w_desc, weight.data_ptr<float>(),
                                                         conv_desc, algo, workspace, workspace_size,
                                                         &beta,
                                                         y_desc, output.data_ptr<float>(),
                                                         bias_desc, bias_opt.value().data_ptr<float>(),
                                                         activation_desc,
                                                         y_desc, output.data_ptr<float>()),
                    "cudnnConvolutionBiasActivationForward failed");
        check_cudnn(cudnnDestroyActivationDescriptor(activation_desc), "cudnnDestroyActivationDescriptor failed");
    } else {
        // No bias: cudnnConvolutionForward
        check_cudnn(cudnnConvolutionForward(cudnn_handle,
                                           &alpha,
                                           x_desc, x.data_ptr<float>(),
                                           w_desc, weight.data_ptr<float>(),
                                           conv_desc, algo, workspace, workspace_size,
                                           &beta,
                                           y_desc, output.data_ptr<float>()),
                    "cudnnConvolutionForward failed");
    }

    // Free resources
    if (workspace_size > 0) { check_cuda(cudaFree(workspace), "cudaFree workspace failed"); }
    check_cudnn(cudnnDestroyConvolutionDescriptor(conv_desc), "cudnnDestroyConvolutionDescriptor failed");
    check_cudnn(cudnnDestroyFilterDescriptor(w_desc), "cudnnDestroyFilterDescriptor failed");
    check_cudnn(cudnnDestroyTensorDescriptor(bias_desc), "cudnnDestroyTensorDescriptor bias failed");
    check_cudnn(cudnnDestroyTensorDescriptor(y_desc), "cudnnDestroyTensorDescriptor y failed");
    check_cudnn(cudnnDestroyTensorDescriptor(x_desc), "cudnnDestroyTensorDescriptor x failed");
    check_cudnn(cudnnDestroy(cudnn_handle), "cudnnDestroy failed");

    return output;
}

// add_bias_forward: out = x + bias, bias shape [C,1,1]
static torch::Tensor add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == x.size(1) && bias.size(1) == 1 && bias.size(2) == 1,
                "bias shape must be [C,1,1]");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(output.data_ptr<float>(), x.data_ptr<float>(), bias.data_ptr<float>(),
                      x.size(0), x.size(1), x.size(2), x.size(3), stream);
    return output;
}

// batch_norm_forward (eval mode)
static torch::Tensor batch_norm_forward(torch::Tensor x, c10::optional<torch::Tensor> weight_opt,
                                        c10::optional<torch::Tensor> bias_opt,
                                        torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == x.size(1), "running_mean shape mismatch");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == x.size(1), "running_var shape mismatch");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight_opt.has_value()) {
        TORCH_CHECK(weight_opt.value().is_cuda() && weight_opt.value().dtype() == torch::kFloat32,
                    "weight must be CUDA float32");
        TORCH_CHECK(weight_opt.value().is_contiguous() && weight_opt.value().dim() == 1 &&
                    weight_opt.value().size(0) == x.size(1), "weight shape mismatch");
        weight_ptr = weight_opt.value().data_ptr<float>();
    }
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda() && bias_opt.value().dtype() == torch::kFloat32,
                    "bias must be CUDA float32");
        TORCH_CHECK(bias_opt.value().is_contiguous() && bias_opt.value().dim() == 1 &&
                    bias_opt.value().size(0) == x.size(1), "bias shape mismatch");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                        weight_ptr, bias_ptr,
                        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
                        static_cast<float>(eps),
                        x.size(0), x.size(1), x.size(2), x.size(3), stream);
    return output;
}

// avg_pool2d_forward
static torch::Tensor avg_pool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "invalid pooling params");

    int64_t out_h = (x.size(2) + 2 * padding - kernel_size) / stride + 1;
    int64_t out_w = (x.size(3) + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::empty({x.size(0), x.size(1), out_h, out_w}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avg_pool2d_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.size(0), x.size(1), x.size(2), x.size(3),
                        (int)kernel_size, (int)stride, (int)padding, stream);
    return output;
}

// sigmoid_forward
static torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_launcher(output.data_ptr<float>(), x.data_ptr<float>(), x.numel(), stream);
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward (cuDNN)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias forward",
          pybind11::arg("x"), pybind11::arg("bias"));
    m.def("batch_norm_forward", &batch_norm_forward, "BatchNorm eval forward",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"));
    m.def("avg_pool2d_forward", &avg_pool2d_forward, "AvgPool2d forward",
          pybind11::arg("x"), pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
