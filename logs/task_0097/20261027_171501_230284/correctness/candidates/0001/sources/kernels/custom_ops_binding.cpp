#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_histc_launcher(
    const float* input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream);

namespace {

// Helper to check cuDNN status
inline void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// Helper to check CUDA status
inline void checkCuda(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(error));
    }
}

// Custom ConvTranspose1d implementation using cuDNN
// Input: [batch, in_channels, length], weight: [in_channels, out_channels, kernel_size], bias: [out_channels]
// stride, padding, output_padding scalar (output_padding applied by adjusting output shape manually)
torch::Tensor custom_conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    TORCH_CHECK(bias.size(0) == weight.size(1), "bias size must match out_channels");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight in_channels must match input channels");

    // Make contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    // Compute output length: (length - 1)*stride - 2*padding + kernel_size + output_padding
    int64_t output_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive");

    // Create output tensor
    auto output = torch::empty({batch, out_channels, output_length}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    checkCudnn(cudnnCreate(&cudnn), "cudnnCreate");
    checkCudnn(cudnnSetStream(cudnn, stream), "cudnnSetStream");

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor input");
    checkCudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor output");
    checkCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias");
    checkCudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor");
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    // Set tensor descriptors: cuDNN expects NCHW format, for 1D we use Nx1xL representation (treat as 2D with height=1)
    // Actually for 1D, cuDNN uses tensor of dim 4 with C, H=1, W=length
    int dims[] = {1, length};  // dummy, but we use 4D descriptor with explicit strides
    // For 1D conv, we set descriptor as NCHW where H=1 and W=length
    int input_nstride = in_channels * length;
    int input_cstride = length;
    int input_hstride = length; // unused for H=1, but must set correctly? Actually cuDNN ignores for H=1
    int input_wstride = 1;
    checkCudnn(cudnnSetTensor4dDescriptor(input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch, in_channels, 1, length),
                "cudnnSetTensor4dDescriptor input");

    int output_nstride = out_channels * output_length;
    int output_cstride = output_length;
    int output_hstride = output_length;
    int output_wstride = 1;
    checkCudnn(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch, out_channels, 1, output_length),
                "cudnnSetTensor4dDescriptor output");

    // Bias descriptor: one bias per output channel
    checkCudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, out_channels, 1, 1),
                "cudnnSetTensor4dDescriptor bias");

    // Filter descriptor: for 1D conv, filter is 3D (in_channels, out_channels, kernel_size)
    // cuDNN expects filter dims: out_channels, in_channels, kernel_h, kernel_w.
    // Since kernel has no spatial height, we set filter as [out_channels, in_channels, 1, kernel_size]
    checkCudnn(cudnnSetFilter4dDescriptor(filter_desc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          out_channels, in_channels, 1, kernel_size),
                "cudnnSetFilter4dDescriptor");

    // Convolution descriptor
    int padA[] = {0, padding};   // height pad = 0, width pad = padding
    int strideA[] = {1, stride}; // height stride = 1, width stride = stride
    int upscaleA[] = {1, 1};
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padA[0], padA[1],
                                               strideA[0], strideA[1],
                                               upscaleA[0], upscaleA[1],
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor");

    // For transposed conv, use cudnnConvolutionBackwardData with appropriate input/output swapped
    // We want to compute output = weight * input transposed, so we use backward data algorithm.
    // Need to find algorithm and workspace size.
    cudnnConvolutionBwdDataAlgo_t algo;
    int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults[1];
    checkCudnn(cudnnFindConvolutionBackwardDataAlgorithm(
                   cudnn,
                   filter_desc,
                   input_desc,   // dy (gradient w.r.t output of forward conv) = input to transposed conv
                   conv_desc,
                   output_desc,  // dx (gradient w.r.t input) = output of transposed conv
                   requestedAlgoCount,
                   &returnedAlgoCount,
                   perfResults),
               "cudnnFindConvolutionBackwardDataAlgorithm");
    if (returnedAlgoCount <= 0) {
        throw std::runtime_error("No convolution backward data algorithm found");
    }
    algo = perfResults[0].algo;

    size_t workspaceSize = 0;
    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
                   cudnn,
                   filter_desc,
                   input_desc,
                   conv_desc,
                   output_desc,
                   algo,
                   &workspaceSize),
               "cudnnGetConvolutionBackwardDataWorkspaceSize");

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        checkCuda(cudaMalloc(&workspace, workspaceSize), "cudaMalloc workspace");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Execute transposed convolution
    checkCudnn(cudnnConvolutionBackwardData(
                   cudnn,
                   &alpha,
                   filter_desc,
                   weight.data_ptr<float>(),
                   input_desc,
                   input.data_ptr<float>(),
                   conv_desc,
                   algo,
                   workspace,
                   workspaceSize,
                   &beta,
                   output_desc,
                   output.data_ptr<float>()),
               "cudnnConvolutionBackwardData");

    // Add bias
    if (bias.numel() > 0) {
        // Use cudnnAddTensor to add bias to each channel
        beta = 1.0f;
        checkCudnn(cudnnAddTensor(cudnn,
                                  &alpha,
                                  bias_desc,
                                  bias.data_ptr<float>(),
                                  &beta,
                                  output_desc,
                                  output.data_ptr<float>()),
                   "cudnnAddTensor");
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    // Check for CUDA errors from cuDNN operations
    checkCuda(cudaGetLastError(), "post cuDNN");

    return output;
}

// Custom histc binding
torch::Tensor custom_histc_forward(
    torch::Tensor input,
    int64_t bins,
    double min_val,
    double max_val)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_histc_launcher(
        input.data_ptr<float>(),
        input.numel(),
        static_cast<int>(bins),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        output.data_ptr<float>(),
        stream
    );

    // Check for errors
    checkCuda(cudaGetLastError(), "histc launch");

    return output;
}

} // anonymous namespace

void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d_forward,
          "Custom ConvTranspose1d using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 2, py::arg("padding") = 1, py::arg("output_padding") = 1);
    m.def("custom_histc", &custom_histc_forward,
          "Custom histogram",
          py::arg("input"), py::arg("bins") = 100,
          py::arg("min") = 0.0, py::arg("max") = 100.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
