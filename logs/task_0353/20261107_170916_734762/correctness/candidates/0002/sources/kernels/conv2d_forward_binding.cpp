// Binding for conv2d_forward operator using cuDNN
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <iostream>
#include <cstring>
#include <memory>
#include "../binding_registry.h"

// Macro for cuDNN error checking
#define CHECK_CUDNN(expr)                                                     \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      throw std::runtime_error(std::string("cuDNN error at ") + __FILE__ +    \
                               ":" + std::to_string(__LINE__) + " - " +       \
                               cudnnGetErrorString(status));                  \
    }                                                                          \
  } while (0)

static torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    // Check input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D [N, C, H, W]");

    // Check weight
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [C_out, C_in, kH, kW]");

    // Bias (optional)
    torch::Tensor bias;
    bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    if (has_bias) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C_out]");
        TORCH_CHECK(bias.size(0) == weight.size(0), "bias size must match weight output channels");
    }

    // Convolution parameters from task_contract: stride=1, padding=1, dilation=1, groups=1
    const int stride_h = 1, stride_w = 1;
    const int pad_h = 1, pad_w = 1;
    const int dilation_h = 1, dilation_w = 1;
    const int groups = 1;

    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Weight dimensions
    int C_out = weight.size(0);
    int C_in_w = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    TORCH_CHECK(C_in == C_in_w, "input channels must match weight input channels");
    TORCH_CHECK(groups == 1, "only groups=1 supported");

    // Output dimensions
    int H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    // Ensure input and weight are contiguous for cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // Output tensor
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));

    // Set tensor descriptors (NCHW format)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));

    // Filter descriptor (C_out, C_in/groups, kH, kW)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in / groups, kH, kW));

    // Convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                               pad_h, pad_w,
                                               stride_h, stride_w,
                                               dilation_h, dilation_w,
                                               CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // Set convolution math type to CUDNN_DEFAULT_MATH
    cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH);

    // Find best algorithm (forward)
    const int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t perf_results[requested_algo_count];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                                     input_desc,
                                                     filter_desc,
                                                     conv_desc,
                                                     output_desc,
                                                     requested_algo_count,
                                                     &returned_algo_count,
                                                     perf_results));
    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    // Workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                        input_desc,
                                                        filter_desc,
                                                        conv_desc,
                                                        output_desc,
                                                        algo,
                                                        &workspace_size));

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc workspace failed: ", cudaGetErrorString(err));
    }

    // Perform convolution
    const float alpha = 1.0f;
    const float beta = 0.0f; // no add to output
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle,
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

    // Add bias if provided (using cudnnAddTensor)
    if (has_bias) {
        // Make bias contiguous
        auto bias_contig = bias.contiguous();
        // Set bias descriptor (1D, NCHW)
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        const float beta_add = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                                   &alpha,
                                   bias_desc,
                                   bias_contig.data_ptr<float>(),
                                   &beta_add,
                                   output_desc,
                                   output.data_ptr<float>()));
    }

    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    // Check CUDA errors after all operations
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error during conv2d_forward: ", cudaGetErrorString(cuda_err));

    return output;
}

static void register_conv2d_forward(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "2D convolution forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none());
}

REGISTER_BINDING(conv2d_forward, register_conv2d_forward);
