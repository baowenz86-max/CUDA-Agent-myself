// Bindings for reflection_pad2d and conv2d using cuDNN.
// This file contains only PyTorch tensor handling and cuDNN API calls,
// no custom CUDA kernels.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <cstdint>

#include "../binding_registry.h"

// ----------------------------------------------------------------------------
// Error checking macros for cuDNN calls
// ----------------------------------------------------------------------------
#define CHECK_CUDNN(expr)                                                       \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      const char* err_str = cudnnGetErrorString(status);                       \
      throw std::runtime_error(std::string("cuDNN error: ") + err_str);        \
    }                                                                          \
  } while (0)

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      const char* err_str = cudaGetErrorString(status);                        \
      throw std::runtime_error(std::string("CUDA error: ") + err_str);         \
    }                                                                          \
  } while (0)

// ----------------------------------------------------------------------------
// extern "C" launcher declarations from custom_ops.cu
// ----------------------------------------------------------------------------
extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int pad,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream);

// ----------------------------------------------------------------------------
// ReflectionPad2d forward
// ----------------------------------------------------------------------------
torch::Tensor reflection_pad2d(torch::Tensor input, int64_t padding) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "reflection_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "reflection_pad2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "reflection_pad2d: input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(padding >= 0, "reflection_pad2d: padding must be >= 0");

    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_h = in_h + 2 * padding;
    int out_w = in_w + 2 * padding;

    // Allocate output tensor (contiguous)
    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    // Get strides (assume contiguous input; but handle arbitrary strides via launcher)
    // The launcher expects strides in elements, not bytes.
    long long in_stride_b = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);

    long long out_stride_b = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        (int)padding,
        in_stride_b,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        stream
    );

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    return output;
}

// ----------------------------------------------------------------------------
// Conv2d forward using cuDNN
// ----------------------------------------------------------------------------
torch::Tensor conv2d_cudnn(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "conv2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "conv2d: input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(weight.is_cuda(), "conv2d: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d: weight must be float32");
    TORCH_CHECK(weight.dim() == 4, "conv2d: weight must be 4D (out_channels, in_channels, kh, kw)");
    TORCH_CHECK(stride > 0, "conv2d: stride must be positive");
    TORCH_CHECK(padding >= 0, "conv2d: padding must be >= 0");

    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "conv2d: bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv2d: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "conv2d: bias must be 1D (out_channels)");
        TORCH_CHECK(bias.size(0) == weight.size(0), "conv2d: bias size must match out_channels");
    }

    // Ensure input and weight are contiguous for cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int batch = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int in_h = input_contig.size(2);
    int in_w = input_contig.size(3);

    int out_channels = weight_contig.size(0);
    int kernel_h = weight_contig.size(2);
    int kernel_w = weight_contig.size(3);

    // Calculate output dimensions
    int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "conv2d: invalid output dimensions");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Set tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input/output tensor descriptors: NCHW format
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        in_channels,
        in_h,
        in_w));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        out_channels,
        out_h,
        out_w));

    // Set filter descriptor: (out_channels, in_channels, kernel_h, kernel_w)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        kernel_h,
        kernel_w));

    // Set convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding,
        padding,
        stride,
        stride,
        1, 1,   // dilation
        CUDNN_CROSS_CORRELATION,  // PyTorch uses cross-correlation
        CUDNN_DATA_FLOAT));

    // Set bias descriptor (only if bias is present)
    if (bias_opt.has_value()) {
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            bias_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            out_channels,
            1,
            1));
    }

    // Find best convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    int requested_algo_count = 1;
    int returned_algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &returned_algo_count,
        &algo));

    // Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
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

    // Set scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc,
        input_contig.data_ptr<float>(),
        filter_desc,
        weight_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>()));

    // Add bias if present
    if (bias_opt.has_value()) {
        float bias_alpha = 1.0f;
        float bias_beta = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(
            cudnn_handle,
            &bias_alpha,
            bias_desc,
            bias.data_ptr<float>(),
            &bias_beta,
            output_desc,
            output.data_ptr<float>()));
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

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d", &reflection_pad2d,
          "ReflectionPad2d forward (CUDA)",
          py::arg("input"),
          py::arg("padding"));

    m.def("conv2d", &conv2d_cudnn,
          "Conv2d forward using cuDNN",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
