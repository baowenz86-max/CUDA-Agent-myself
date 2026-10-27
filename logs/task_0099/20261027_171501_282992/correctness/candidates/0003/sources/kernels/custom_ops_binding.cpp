// Binding for sqrt, round, and conv_transpose2d custom operations.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <vector>
#include <iostream>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (to be implemented later)
extern "C" void sqrt_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* input_shape,
    int64_t ndims,
    cudaStream_t stream);

extern "C" void round_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* input_shape,
    int64_t ndims,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Check CUDA error
static void checkCudaError(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Check cuDNN error
static void checkCudnnError(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// Tensor wrapper for sqrt
at::Tensor sqrt_forward(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto output = at::empty_like(input);
    if (input.numel() == 0) return output;

    std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> input_strides(input.strides().begin(), input.strides().end());
    int64_t total = input.numel();

    cudaStream_t stream = getCurrentStream();
    sqrt_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        total,
        input_strides.data(),
        input_shape.data(),
        input.dim(),
        stream
    );
    checkCudaError(cudaGetLastError(), "sqrt_launcher failed");
    return output;
}

// Tensor wrapper for round
at::Tensor round_forward(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto output = at::empty_like(input);
    if (input.numel() == 0) return output;

    std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> input_strides(input.strides().begin(), input.strides().end());
    int64_t total = input.numel();

    cudaStream_t stream = getCurrentStream();
    round_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        total,
        input_strides.data(),
        input_shape.data(),
        input.dim(),
        stream
    );
    checkCudaError(cudaGetLastError(), "round_launcher failed");
    return output;
}

// cuDNN-based ConvTranspose2d forward
at::Tensor conv_transpose2d_forward(
    at::Tensor input,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Weight must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(stride > 0 && padding >= 0 && output_padding >= 0, "Invalid conv parameters");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->dtype() == at::kFloat && bias->dim() == 1, "Bias must be CUDA float32 1D");
        TORCH_CHECK(bias->size(0) == weight.size(1), "Bias size must match out_channels");
    }

    // Ensure contiguous tensors for cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    at::Tensor bias_contig;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
    }

    // Dimensions
    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t H_in = input_contig.size(2);
    int64_t W_in = input_contig.size(3);
    int64_t C_out = weight_contig.size(1);
    int64_t kH = weight_contig.size(2);
    int64_t kW = weight_contig.size(3);

    // Output dimensions
    int64_t H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int64_t W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto output = at::empty({N, C_out, H_out, W_out}, input_contig.options());

    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle), "cudnnCreate");
    cudaStream_t stream = getCurrentStream();
    checkCudnnError(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream");

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor input");
    checkCudnnError(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor output");
    checkCudnnError(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias");

    // Set tensor descriptors; NCHW layout
    checkCudnnError(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H_in, W_in), "cudnnSetTensor4dDescriptor input");
    checkCudnnError(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out), "cudnnSetTensor4dDescriptor output");
    checkCudnnError(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1), "cudnnSetTensor4dDescriptor bias");

    cudnnFilterDescriptor_t filter_desc;
    checkCudnnError(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor");
    checkCudnnError(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight_contig.size(0), weight_contig.size(1), kH, kW), "cudnnSetFilter4dDescriptor");

    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");
    checkCudnnError(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");
    // Note: For transposed convolution, use cudnnConvolutionBackwardData.

    // Use cudnnConvolutionBackwardData for transposed conv: output = input *^T weight
    float alpha = 1.0f, beta = 0.0f;
    checkCudnnError(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        nullptr, // workspace
        0,       // workspace size
        &beta,
        output_desc,
        output.data_ptr<float>()
    ), "cudnnConvolutionBackwardData");

    // Add bias if present
    if (bias.has_value()) {
        float beta_bias = 1.0f;
        checkCudnnError(cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias_contig.data_ptr<float>(), &beta_bias, output_desc, output.data_ptr<float>()), "cudnnAddTensor");
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    checkCudaError(cudaGetLastError(), "conv_transpose2d_forward failed");
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("sqrt_forward", &sqrt_forward, "Element-wise square root", py::arg("input"));
    m.def("round_forward", &round_forward, "Element-wise round to nearest even", py::arg("input"));
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "2D transposed convolution (cuDNN)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride") = 1, py::arg("padding") = 0, py::arg("output_padding") = 0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
