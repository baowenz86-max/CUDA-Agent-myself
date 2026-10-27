#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Declare CUDA launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void pixel_shuffle_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int upscale_factor,
    cudaStream_t stream);

namespace {

// Helper to check cuDNN status
void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// Make tensor contiguous if not already
void ensure_contiguous(torch::Tensor& t) {
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
}

// cuDNN-based ConvTranspose2d forward
// Input: x [N, C_in, H_in, W_in]
// Weight: weight [C_in, C_out, kH, kW] (PyTorch ConvTranspose2d weight layout)
// Bias: bias [C_out] or empty optional
// Returns output [N, C_out, H_out, W_out]
torch::Tensor deconv_forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (C_in, C_out, kH, kW)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "Only float32 bias supported");

    // Make contiguous copies if needed
    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    auto bias_contig = (bias.defined() && bias.numel() > 0) ? (bias.is_contiguous() ? bias : bias.contiguous()) : torch::Tensor();

    int64_t N = x_contig.size(0);
    int64_t C_in = x_contig.size(1);
    int64_t H_in = x_contig.size(2);
    int64_t W_in = x_contig.size(3);
    int64_t C_out = weight_contig.size(1);
    int64_t kH = weight_contig.size(2);
    int64_t kW = weight_contig.size(3);

    TORCH_CHECK(C_in == weight_contig.size(0), "Input channels must match weight in_channels");
    TORCH_CHECK(kH == kernel_size && kW == kernel_size, "Kernel size mismatch");

    int64_t H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int64_t W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions");

    // Allocate output
    auto output = torch::empty({N, C_out, H_out, W_out}, x_contig.options());

    // cuDNN handles
    cudnnHandle_t cudnn_handle;
    checkCudnn(cudnnCreate(&cudnn_handle), "cudnnCreate");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    checkCudnn(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream");

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x_desc");
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y_desc");
    checkCudnn(cudnnCreateTensorDescriptor(&b_desc), "cudnnCreateTensorDescriptor b_desc");

    // Set tensor descriptors with NCHW format
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         N, C_in, H_in, W_in), "cudnnSetTensor4dDescriptor x");
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         N, C_out, H_out, W_out), "cudnnSetTensor4dDescriptor y");
    checkCudnn(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         1, C_out, 1, 1), "cudnnSetTensor4dDescriptor bias");

    // Filter descriptor (weight layout: [C_in, C_out, kH, kW])
    cudnnFilterDescriptor_t w_desc;
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor");
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                         C_in, C_out, kH, kW), "cudnnSetFilter4dDescriptor");

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padding, padding,
                                               stride, stride,
                                               1, 1, // dilation
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");

    // Choose algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    int requested_algo_count = 1;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    checkCudnn(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_handle, w_desc, x_desc, conv_desc, y_desc, 1, &requested_algo_count, &perf_results),
        "cudnnFindConvolutionBackwardDataAlgorithm");
    algo = perf_results.algo;

    // Workspace size
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle, w_desc, x_desc, conv_desc, y_desc, algo, &workspace_size),
        "cudnnGetConvolutionBackwardDataWorkspaceSize");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_status == cudaSuccess, "cudaMalloc failed for cuDNN workspace");
    }

    // Perform convolution backward data (equivalent to transposed convolution forward)
    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc, weight_contig.data_ptr<float>(),
        x_desc, x_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>()),
        "cudnnConvolutionBackwardData");

    // Add bias if provided
    if (bias_contig.defined() && bias_contig.numel() > 0) {
        float beta_bias = 1.0f;
        checkCudnn(cudnnAddTensor(cudnn_handle,
                                  &alpha,
                                  b_desc, bias_contig.data_ptr<float>(),
                                  &beta_bias,
                                  y_desc, output.data_ptr<float>()),
                   "cudnnAddTensor bias");
    }

    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

torch::Tensor pixel_shuffle_forward_impl(const torch::Tensor& x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C*r^2, H, W)");

    auto x_contig = x.is_contiguous() ? x : x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C_in = x_contig.size(1);
    int64_t H_in = x_contig.size(2);
    int64_t W_in = x_contig.size(3);
    int64_t r = upscale_factor;

    TORCH_CHECK(C_in % (r * r) == 0, "Channel dimension must be divisible by r^2");
    int64_t C_out = C_in / (r * r);
    int64_t H_out = H_in * r;
    int64_t W_out = W_in * r;

    auto output = torch::empty({N, C_out, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pixel_shuffle_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_out, H_in, W_in, H_out, W_out,
        static_cast<int>(upscale_factor),
        stream);

    return output;
}

torch::Tensor relu_forward_impl(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream);

    return output;
}

} // anonymous namespace

// Python bindings
torch::Tensor deconv_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    return deconv_forward_impl(x, weight, bias, kernel_size, stride, padding, output_padding);
}

torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    return pixel_shuffle_forward_impl(x, upscale_factor);
}

torch::Tensor relu_forward(torch::Tensor x) {
    return relu_forward_impl(x);
}

void register_custom_ops(pybind11::module& m) {
    m.def("deconv_forward", &deconv_forward, "ConvTranspose2d forward using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward, "PixelShuffle forward",
          py::arg("x"), py::arg("upscale_factor"));
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
