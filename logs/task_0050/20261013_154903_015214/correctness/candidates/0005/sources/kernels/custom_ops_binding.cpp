#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Helper to check cuDNN status
#define CHECK_CUDNN(status) \
    do { \
        cudnnStatus_t _status = (status); \
        TORCH_CHECK(_status == CUDNN_STATUS_SUCCESS, \
            "cuDNN error: ", cudnnGetErrorString(_status)); \
    } while (0)

// Utility to convert torch dtype to cudnnDataType_t
cudnnDataType_t getCudnnDataType(torch::Dtype dtype) {
    if (dtype == torch::kFloat32) return CUDNN_DATA_FLOAT;
    if (dtype == torch::kFloat16) return CUDNN_DATA_HALF;
    if (dtype == torch::kFloat64) return CUDNN_DATA_DOUBLE;
    TORCH_CHECK(false, "Unsupported dtype for cuDNN convolution");
}

// Function to perform convolution using cuDNN
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels/groups, kh, kw)");
    TORCH_CHECK(input.dtype() == weight.dtype(), "input and weight must have same dtype");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Bias optional checks
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias_tensor.size(0) == weight.size(0), "bias size must match out_channels");
        TORCH_CHECK(bias_tensor.dtype() == input.dtype(), "bias dtype must match input");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
    }

    // Dimension extraction
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t K = weight.size(0); // out_channels
    int64_t C_per_group = weight.size(1);
    int64_t kh = weight.size(2);
    int64_t kw = weight.size(3);

    TORCH_CHECK(C == C_per_group * groups, "input channels must equal groups * C_per_group");
    TORCH_CHECK(K % groups == 0, "out_channels must be divisible by groups");

    // Compute output dimensions (as PyTorch does with padding and dilation)
    int64_t out_h = (H + 2 * padding_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * padding_w - dilation_w * (kw - 1) - 1) / stride_w + 1;

    TORCH_CHECK(out_h > 0 && out_w > 0, "output size must be positive");

    // Allocate output tensor
    auto output = torch::empty({N, K, out_h, out_w}, input.options());

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

    cudnnDataType_t data_type = getCudnnDataType(input.dtype());

    // Set input descriptor: NCHW
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        data_type,
        N, C, H, W
    ));

    // Set output descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        data_type,
        N, K, out_h, out_w
    ));

    // Set bias descriptor (for bias add later, if needed)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        data_type,
        1, K, 1, 1
    ));

    // Set filter descriptor: K, C_per_group, kh, kw
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        data_type,
        CUDNN_TENSOR_NCHW,
        K, C_per_group, kh, kw
    ));

    // Set convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
        data_type
    ));

    if (groups > 1) {
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, groups));
    }

    // Determine algorithm
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[requested_algo_count];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &returned_algo_count,
        perf_results
    ));
    TORCH_CHECK(returned_algo_count > 0, "No cuDNN algorithm found");
    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    // Determine workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size
    ));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(workspace != nullptr, "Failed to allocate workspace");
    }

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input.data_ptr(),
        filter_desc, weight.data_ptr(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        output_desc, output.data_ptr()
    ));

    // Add bias if present
    if (bias_tensor.defined()) {
        float beta_bias = 1.0f; // accumulate to output
        CHECK_CUDNN(cudnnAddTensor(
            cudnn_handle,
            &alpha,
            bias_desc, bias_tensor.data_ptr(),
            &beta_bias,
            output_desc, output.data_ptr()
        ));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    // Check for kernel errors? The stream will handle async errors; we can sync if needed
    // but typically we rely on PyTorch's error checking later.
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def(
        "conv2d_forward",
        &conv2d_forward,
        "Perform 2D convolution using cuDNN",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("padding_h") = 0,
        py::arg("padding_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::arg("groups") = 1
    );
}

REGISTER_BINDING(custom_ops, register_custom_ops);
