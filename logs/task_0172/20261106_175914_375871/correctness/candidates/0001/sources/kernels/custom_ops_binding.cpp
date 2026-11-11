// PyTorch bindings for custom ops.
// File: kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// -----------------------------------------------------------------------------
// Forward declarations for launchers defined in custom_ops.cu
// -----------------------------------------------------------------------------

// triu_last2d: applies torch.triu to the last two dimensions for each batch/channel.
extern "C" void triu_last2d_launcher(
    const float* x,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    long long stride_batch,
    long long stride_channel,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Helper: check CUDA error
// -----------------------------------------------------------------------------
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

static void checkCudnnError(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// -----------------------------------------------------------------------------
// cuDNN conv2d implementation
// -----------------------------------------------------------------------------

static torch::Tensor conv2d_cudnn(
    const torch::Tensor& input,
    const torch::Tensor& weight
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "conv2d input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d weight must be float32");
    TORCH_CHECK(input.dim() == 4, "conv2d input must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "conv2d weight must be 4D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int k_h = weight.size(2);
    int k_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels, "weight has wrong in_channels");

    // Fixed convolution parameters from reference: stride=(1,1), padding=(1,2), dilation=(2,1)
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 2;
    int dilation_h = 2, dilation_w = 1;

    // Compute output dimensions (same formula as PyTorch)
    int out_h = (in_h + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;

    // Create output tensor (contiguous)
    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    checkCudnnError(cudnnCreate(&cudnn), "cudnnCreate");
    checkCudnnError(cudnnSetStream(cudnn, stream), "cudnnSetStream");

    // Set tensor descriptors (NCHW format)
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor");
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor");
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor");
    checkCudnnError(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    checkCudnnError(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, in_channels, in_h, in_w), "cudnnSetTensor4dDescriptor x");
    checkCudnnError(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_channels, out_h, out_w), "cudnnSetTensor4dDescriptor y");
    checkCudnnError(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_h, k_w), "cudnnSetFilter4dDescriptor");
    checkCudnnError(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ), "cudnnSetConvolution2dDescriptor");

    // Find an appropriate convolution algorithm
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    checkCudnnError(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        x_desc, w_desc, conv_desc, y_desc,
        requested_algo_count,
        &returned_algo_count,
        perf_results
    ), "cudnnGetConvolutionForwardAlgorithm_v7");
    if (returned_algo_count == 0) {
        throw std::runtime_error("cuDNN returned no convolution algorithm");
    }
    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    // Query workspace size
    size_t workspace_size = 0;
    checkCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        x_desc, w_desc, conv_desc, y_desc,
        algo,
        &workspace_size
    ), "cudnnGetConvolutionForwardWorkspaceSize");

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace");
    }

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    checkCudnnError(cudnnConvolutionForward(
        cudnn,
        &alpha,
        x_desc, input.data_ptr<float>(),
        w_desc, weight.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>()
    ), "cudnnConvolutionForward");

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
    checkCudaError(cudaGetLastError(), "cudaGetLastError after conv");

    return output;
}

// -----------------------------------------------------------------------------
// PyTorch wrapper functions (exposed to Python)
// -----------------------------------------------------------------------------

// triu_last2d
static torch::Tensor triu_last2d(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "triu_last2d input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "triu_last2d input must be float32");
    TORCH_CHECK(x.dim() >= 2, "triu_last2d input must have at least 2 dimensions");

    int batch = 1;
    int channels = 1;
    int64_t height = 1;
    int64_t width = 1;
    if (x.dim() == 2) {
        batch = 1;
        channels = 1;
        height = x.size(0);
        width = x.size(1);
    } else if (x.dim() == 3) {
        batch = 1;
        channels = x.size(0);
        height = x.size(1);
        width = x.size(2);
    } else if (x.dim() >= 4) {
        // Usually NCHW: all dims before last two are treated as batch*channels for the mask.
        batch = 1;
        channels = 1;
        for (int i = 0; i < x.dim() - 2; ++i) {
            // flatten all leading dimensions into channels? But we need to preserve strides.
            // The launcher expects batch and channels separately. We can pass batch = number of leading groups,
            // channels = 1 if we squeeze. But better: follow common NCHW case from task.
            // We'll assume the standard NCHW for now; general case requires more complex handling.
            // For safety, we handle only 4D here, throw otherwise.
        }
        TORCH_CHECK(x.dim() == 4, "triu_last2d only supports 4D (NCHW) tensors in this binding");
        batch = x.size(0);
        channels = x.size(1);
        height = x.size(2);
        width = x.size(3);
    } else {
        TORCH_CHECK(false, "Unsupported dim");
    }

    // Output tensor: same shape as input, contiguous? Actually torch.triu returns a tensor with same strides as input
    // (view), so we should preserve strides. But our launcher writes elementwise assuming same strides as input,
    // which is correct if we allocate a contiguous tensor and copy? Wait, the launcher assumes output uses same strides as input.
    // To avoid mismatch, we can allocate output with the same strides as input by using empty_like, which copies strides only if memory_format is preserved.
    // torch::empty_like(x) will produce contiguous unless x has non-standard memory format? It does not copy strides by default; it uses same options but contiguous.
    // Better: use torch::empty_strided(x.sizes(), x.strides(), x.options()) to preserve strides.
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_last2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        (int)height,
        (int)width,
        (long long)x.stride(0),
        (long long)x.stride(1),
        (long long)x.stride(2),
        (long long)x.stride(3),
        stream
    );
    checkCudaError(cudaGetLastError(), "triu_last2d_launcher");
    return output;
}

// conv2d wrapper
static torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight) {
    // Input may be non-contiguous due to triu_last2d preserving strides; cuDNN requires contiguous input.
    // We make it contiguous if needed.
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    return conv2d_cudnn(input, weight);
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------

void register_custom_ops(pybind11::module& m) {
    m.def("triu_last2d", &triu_last2d, "Apply torch.triu to last two dimensions", py::arg("x"));
    m.def("conv2d", &conv2d, "2D convolution using cuDNN", py::arg("input"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
