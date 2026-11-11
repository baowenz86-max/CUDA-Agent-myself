// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_channel_last_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int numel,
    float negative_slope,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Helper to check CUDA errors
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Helper to check cuDNN errors
static void checkCudnnError(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// Softmax wrapper
torch::Tensor softmax_channel_last_placeholder(torch::Tensor x, int dim) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 supported");

    // Ensure channels_last layout (NHWC). If not, convert (but contract says channels_last, so should be okay)
    auto x_cl = x.to(torch::MemoryFormat::ChannelsLast);
    if (!x.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        x_cl = x.contiguous();  // fallback to contiguous NCHW? But launcher expects NHWC.
    }

    auto sizes = x_cl.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto output = torch::empty_like(x_cl, torch::MemoryFormat::ChannelsLast);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_channel_last_launcher(
        output.data_ptr<float>(),
        x_cl.data_ptr<float>(),
        N, C, H, W,
        stream
    );
    checkCudaError(cudaGetLastError(), "softmax_channel_last_launcher failed");

    return output;
}

// LeakyReLU wrapper
torch::Tensor leaky_relu_placeholder(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Works on contiguous data; numel is same regardless of memory format
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        static_cast<float>(negative_slope),
        stream
    );
    checkCudaError(cudaGetLastError(), "leaky_relu_launcher failed");

    return output;
}

// Add bias wrapper
torch::Tensor add_bias_placeholder(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    TORCH_CHECK(bias.numel() == C, "Bias numel must match channels");

    // Ensure channels_last layout for launcher
    auto x_cl = x.to(torch::MemoryFormat::ChannelsLast);
    auto output = torch::empty_like(x_cl, torch::MemoryFormat::ChannelsLast);

    // bias is shape [C,1,1], but launcher expects contiguous array of C floats
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        output.data_ptr<float>(),
        x_cl.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N, C, H, W,
        stream
    );
    checkCudaError(cudaGetLastError(), "add_bias_launcher failed");

    return output;
}

// Conv2d using cuDNN
torch::Tensor conv2d_cudnn_placeholder(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (O,I,kH,kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    TORCH_CHECK(weight.size(1) == C_in, "weight input channels mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must equal out channels");

    int pad_h = static_cast<int>(padding);
    int pad_w = static_cast<int>(padding);
    int stride_h = static_cast<int>(stride);
    int stride_w = static_cast<int>(stride);

    // Calculate output dimensions
    int H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    // Create output tensor (channels_last as per contract)
    auto output = torch::empty({N, C_out, H_out, W_out}, x.options().memory_format(torch::MemoryFormat::ChannelsLast));

    // cuDNN requires contiguous NCHW for standard operations, but can handle NHWC with appropriate tensor format.
    // We'll use cuDNN with tensor format NHWC for input and output, weight as OIHW (contiguous).
    // Ensure tensors are in the expected memory layout.
    auto x_cl = x.to(torch::MemoryFormat::ChannelsLast);
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    checkCudnnError(cudnnCreate(&cudnn), "cudnnCreate failed");

    // Set stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    checkCudnnError(cudnnSetStream(cudnn, stream), "cudnnSetStream failed");

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc), "create x desc failed");
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc), "create y desc failed");
    checkCudnnError(cudnnCreateTensorDescriptor(&bias_desc), "create bias desc failed");
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc), "create w desc failed");
    checkCudnnError(cudnnCreateConvolutionDescriptor(&conv_desc), "create conv desc failed");

    // Set tensor descriptors with NHWC layout
    // For NHWC, strides: N*H*W*C, W*C, C, 1
    // cudnnSetTensor4dDescriptor expects N,C,H,W with strides, but for NHWC we need to specify strides accordingly.
    // To avoid complex stride calc, we can use cudnnSetTensor4dDescriptorEx with explicit strides.
    // But simpler: convert x to NCHW contiguous, perform conv, then convert output back to NHWC. This adds overhead but safe.
    // Considering performance, we can use cudnn's NHWC support by setting tensor format to CUDNN_TENSOR_NHWC (cudnnSetTensor4dDescriptor with format).
    // cudnnSetTensor4dDescriptor with CUDNN_TENSOR_NHWC is available in cuDNN 7+.

    // Use CUDNN_TENSOR_NHWC for input and output
    checkCudnnError(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        N, C_in, H_in, W_in
    ), "set x desc failed");

    checkCudnnError(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        N, C_out, H_out, W_out
    ), "set y desc failed");

    // Bias descriptor: shape 1,C,1,1 (broadcast across spatial and batch)
    checkCudnnError(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,  // Bias is 1D, but we can set as NCHW with C_out,1,1
        CUDNN_DATA_FLOAT,
        1, C_out, 1, 1
    ), "set bias desc failed");

    // Filter descriptor: OIHW (C_out, C_in, kH, kW)
    checkCudnnError(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,  // Filter format OIHW, stored as NCHW-like but actually OIhw
        C_out, C_in, kH, kW
    ), "set filter desc failed");

    // Convolution descriptor
    checkCudnnError(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w,
        stride_h, stride_w,
        1, 1, // dilation
        CUDNN_CROSS_CORRELATION, // PyTorch conv2d uses cross-correlation
        CUDNN_DATA_FLOAT
    ), "set conv desc failed");

    // Set math type to default (allows TF32 if available)
    checkCudnnError(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH), "set math type failed");

    // Find algorithm
    cudnnConvolutionFwdAlgo_t algo;
    int requestedAlgoCount = 1;
    int returnedAlgoCount = -1;
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];
    checkCudnnError(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        1,
        &returnedAlgoCount,
        perfResults
    ), "find algo failed");
    if (returnedAlgoCount <= 0) {
        throw std::runtime_error("No suitable cuDNN algorithm found");
    }
    algo = perfResults[0].algo;

    // Allocate workspace
    size_t workspace_size = 0;
    checkCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_size
    ), "get workspace size failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace failed");
    }

    // Perform convolution with bias
    float alpha = 1.0f;
    float beta = 0.0f;  // For bias addition, we can use beta=1.0 after zero-initialization, or use cudnnConvolutionBiasActivationForward.
    // Simpler: do convolution with beta=0 (output = conv), then add bias using our add_bias kernel or cudnnAddTensor.
    // But we have add_bias_launcher for that. However, to avoid extra kernel, we can use cudnnConvolutionBiasActivationForward.
    // We'll use cudnnConvolutionForward first, then add bias using cudnnAddTensor (or our custom add_bias). But to keep consistent with model_new.py, we already have add_bias_placeholder for the extra bias after conv. Wait: in model_new.py, conv2d_cudnn_placeholder takes x, weight, bias (conv.bias) and returns conv result with conv.bias added. Then add_bias_placeholder adds extra bias. So conv2d should include conv.bias.
    // So we need to add bias in conv2d. Options:
    // 1. Use cudnnConvolutionBiasActivationForward with activation identity.
    // 2. Do convolution, then add bias via separate kernel call.
    // We'll use cudnnConvolutionBiasActivationForward for efficiency.

    // Set activation descriptor for identity (no activation)
    cudnnActivationDescriptor_t activation_desc;
    checkCudnnError(cudnnCreateActivationDescriptor(&activation_desc), "create activation desc failed");
    checkCudnnError(cudnnSetActivationDescriptor(
        activation_desc,
        CUDNN_ACTIVATION_IDENTITY,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0
    ), "set activation desc failed");

    checkCudnnError(cudnnConvolutionBiasActivationForward(
        cudnn,
        &alpha,
        x_desc, x_cl.data_ptr<float>(),
        w_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>(),
        bias_desc, bias_contig.data_ptr<float>(),
        activation_desc,
        y_desc, output.data_ptr<float>()
    ), "conv bias activation failed");

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    checkCudaError(cudaGetLastError(), "conv2d failed");

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("softmax_channel_last_placeholder", &softmax_channel_last_placeholder,
          "Softmax along channel dim for channels_last tensor",
          pybind11::arg("x"), pybind11::arg("dim"));
    m.def("leaky_relu_placeholder", &leaky_relu_placeholder,
          "LeakyReLU elementwise",
          pybind11::arg("x"), pybind11::arg("negative_slope"));
    m.def("conv2d_cudnn_placeholder", &conv2d_cudnn_placeholder,
          "Conv2d with cuDNN, includes bias",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("add_bias_placeholder", &add_bias_placeholder,
          "Add bias broadcast",
          pybind11::arg("x"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
