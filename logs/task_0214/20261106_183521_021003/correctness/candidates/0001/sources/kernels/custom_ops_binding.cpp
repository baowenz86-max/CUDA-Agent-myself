// PyTorch binding for custom operators
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Helper function to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Check if tensor is CUDA and float32
static void check_cuda_float32(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// ReflectionPad2d forward
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int pad = static_cast<int>(padding);
    int outH = H + 2 * pad;
    int outW = W + 2 * pad;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = getCurrentStream();
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        pad,
        stream);

    return output;
}

// Sigmoid forward
torch::Tensor sigmoid_forward(torch::Tensor input) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");

    auto output = torch::empty_like(input);
    int total = static_cast<int>(input.numel());

    cudaStream_t stream = getCurrentStream();
    sigmoid_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

// Conv2d forward using cuDNN
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    check_cuda_float32(input, "input");
    check_cuda_float32(weight, "weight");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int K = weight.size(0);
    int C_per_group = weight.size(1);
    int R = weight.size(2);
    int S = weight.size(3);

    int sH = static_cast<int>(stride);
    int sW = static_cast<int>(stride);
    int pH = static_cast<int>(padding);
    int pW = static_cast<int>(padding);
    int dH = static_cast<int>(dilation);
    int dW = static_cast<int>(dilation);
    int g = static_cast<int>(groups);

    TORCH_CHECK(C == C_per_group * g, "input channels must equal weight[1]*groups");
    TORCH_CHECK(K % g == 0, "output channels must be divisible by groups");

    int outH = (H + 2 * pH - dH * (R - 1) - 1) / sH + 1;
    int outW = (W + 2 * pW - dW * (S - 1) - 1) / sW + 1;

    auto output = torch::empty({N, K, outH, outW}, input.options());

    // Initialize cuDNN
    cudnnHandle_t cudnn_handle;
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));

    cudaStream_t stream = getCurrentStream();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));

    // Input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C, H, W));

    // Output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, K, outH, outW));

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        K, C_per_group, R, S));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pH, pW,       // padding
        sH, sW,       // stride
        dH, dW,       // dilation
        CUDNN_CROSS_CORRELATION, // PyTorch convolution is cross-correlation
        CUDNN_DATA_FLOAT));
    if (g > 1) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, g));
    }

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    // Workspace
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc,
        input.data_ptr<float>(),
        filter_desc,
        weight.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>()));

    // Add bias
    cudnnTensorDescriptor_t bias_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, K, 1, 1));

    CUDNN_CHECK(cudnnAddTensor(
        cudnn_handle,
        &alpha,
        bias_desc,
        bias.data_ptr<float>(),
        &alpha,
        output_desc,
        output.data_ptr<float>()));

    // Cleanup
    if (workspace) cudaFree(workspace);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));

    return output;
}

// Add bias forward
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float32(input, "input");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (C,1,1)");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias spatial dims must be 1");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentStream();
    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

// Register all bindings
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward",
          py::arg("input"));
    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward with cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
