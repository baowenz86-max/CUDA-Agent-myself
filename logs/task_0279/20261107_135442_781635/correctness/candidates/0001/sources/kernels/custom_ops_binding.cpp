#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_CUDNN(x) TORCH_CHECK((x) == CUDNN_STATUS_SUCCESS, "cuDNN error: " + std::string(cudnnGetErrorString(x)))

// Forward declaration of normalize launcher from custom_ops.cu
extern "C" void normalize_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    cudaStream_t stream);

// Helper to check cuDNN availability and create handle
cudnnHandle_t getCudnnHandle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUDNN(cudnnCreate(&handle));
    }
    return handle;
}

// Helper to get current CUDA stream
cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper to convert torch::Tensor to raw pointer
template <typename T>
T* data_ptr(torch::Tensor& t) {
    return t.data_ptr<T>();
}

// Helper to convert int64_t dims to int for cuDNN (assuming small)
std::vector<int> toIntVec(torch::IntArrayRef dims) {
    std::vector<int> result;
    result.reserve(dims.size());
    for (auto d : dims) {
        result.push_back(static_cast<int>(d));
    }
    return result;
}

// ConvTranspose1d implementation using cuDNN
torch::Tensor conv_transpose1d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Input checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_FLOAT32(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    }
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    // Compute output dimensions
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t kernel_size = weight.size(2);
    int64_t C_out = weight.size(1);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");
    TORCH_CHECK(weight.size(0) == C_in, "weight in_channels mismatch");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == C_out, "bias size mismatch with out_channels");
    }

    // Allocate output tensor
    auto output = torch::empty({N, C_out, L_out}, input.options());

    // Make input and weight contiguous if not already (cuDNN requires contiguous?)
    // Actually cuDNN can handle strides with some constraints, but for simplicity we use contiguous.
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto output_contig = output; // output is already contiguous

    // Set up cuDNN tensor descriptors
    cudnnHandle_t handle = getCudnnHandle();
    cudaStream_t stream = getCurrentStream();
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors: data layout NCHW (for 1D, treat length as width, height=1)
    // We'll use CUDNN_TENSOR_NCHW with dims {N, C_in, 1, L_in} for input and {N, C_out, 1, L_out} for output
    std::vector<int> input_dims = {static_cast<int>(N), static_cast<int>(C_in), 1, static_cast<int>(L_in)};
    std::vector<int> output_dims = {static_cast<int>(N), static_cast<int>(C_out), 1, static_cast<int>(L_out)};
    std::vector<int> filter_dims = {static_cast<int>(C_in), static_cast<int>(C_out), 1, static_cast<int>(kernel_size)};
    std::vector<int> input_strides = {static_cast<int>(C_in * 1 * L_in), static_cast<int>(1 * L_in), static_cast<int>(L_in), 1};
    std::vector<int> output_strides = {static_cast<int>(C_out * 1 * L_out), static_cast<int>(1 * L_out), static_cast<int>(L_out), 1};
    std::vector<int> bias_dims = {1, static_cast<int>(C_out), 1, 1};
    std::vector<int> bias_strides = {static_cast<int>(C_out), 1, 1, 1};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, input_dims.data(), input_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4, output_dims.data(), output_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 4, bias_dims.data(), bias_strides.data()));
    // Filter descriptor: for cuDNN filter tensor is KCRS (output_channels, input_channels, height, width)
    // However for transposed conv it's input channels, output channels? Check: cuDNN uses filter layout for convolution as (K, C, R, S) where K=output channels, C=input channels.
    // For conv_transpose, it's the same filter but the operation is backward data, so we set filter dims accordingly: (C_out, C_in, 1, kernel_size)
    std::vector<int> filter_dims_cudnn = {static_cast<int>(C_out), static_cast<int>(C_in), 1, static_cast<int>(kernel_size)};
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims_cudnn.data()));

    // Convolution descriptor for transposed conv (backward data): use padding, stride, dilation=1
    // For cuDNN transposed convolution via cudnnConvolutionBackwardData, we set conv_desc for forward convolution that would produce this backward.
    // Therefore we set padding=padding, stride=stride, and use the transposed operation.
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 2, &padding_int, &stride_int, &dilation_int, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Allocate workspace
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, input_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    // Execute transposed convolution (backward data)
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc, weight_contig.data_ptr<float>(),
        input_desc, input_contig.data_ptr<float>(),
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        workspace, workspace_size,
        &beta,
        output_desc, output_contig.data_ptr<float>()));

    // Add bias if present
    if (bias.has_value()) {
        CHECK_CUDNN(cudnnAddTensor(handle, &alpha, bias_desc, bias.value().contiguous().data_ptr<float>(), &alpha, output_desc, output_contig.data_ptr<float>()));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

// Normalize implementation using launcher from .cu
torch::Tensor normalize(torch::Tensor input, double p, int64_t dim) {
    CHECK_CUDA(input);
    CHECK_FLOAT32(input);
    TORCH_CHECK(input.dim() == 3, "normalize input must be 3D (N, C, L)");
    TORCH_CHECK(dim == 1, "normalize only supports dim=1");
    TORCH_CHECK(p == 2.0, "normalize only supports p=2");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = getCurrentStream();

    normalize_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(L),
        stream);

    return output;
}

// Conv1d implementation using cuDNN
torch::Tensor conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t groups) {
    // Input checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_FLOAT32(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    }
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels/groups, kernel_size)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t kernel_size = weight.size(2);
    int64_t C_out = weight.size(0);
    int64_t L_out = (L_in + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(L_out > 0, "output length must be positive");
    TORCH_CHECK(weight.size(1) * groups == C_in, "weight in_channels/groups mismatch");
    TORCH_CHECK(C_out % groups == 0, "out_channels must be divisible by groups");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == C_out, "bias size mismatch with out_channels");
    }

    auto output = torch::empty({N, C_out, L_out}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto output_contig = output;

    cudnnHandle_t handle = getCudnnHandle();
    cudaStream_t stream = getCurrentStream();
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    std::vector<int> input_dims = {static_cast<int>(N), static_cast<int>(C_in), 1, static_cast<int>(L_in)};
    std::vector<int> output_dims = {static_cast<int>(N), static_cast<int>(C_out), 1, static_cast<int>(L_out)};
    std::vector<int> filter_dims = {static_cast<int>(C_out), static_cast<int>(C_in / groups), 1, static_cast<int>(kernel_size)};
    std::vector<int> input_strides = {static_cast<int>(C_in * 1 * L_in), static_cast<int>(1 * L_in), static_cast<int>(L_in), 1};
    std::vector<int> output_strides = {static_cast<int>(C_out * 1 * L_out), static_cast<int>(1 * L_out), static_cast<int>(L_out), 1};
    std::vector<int> bias_dims = {1, static_cast<int>(C_out), 1, 1};
    std::vector<int> bias_strides = {static_cast<int>(C_out), 1, 1, 1};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, input_dims.data(), input_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4, output_dims.data(), output_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 4, bias_dims.data(), bias_strides.data()));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims.data()));

    int padA[2] = {static_cast<int>(padding), 0};
    int strideA[2] = {static_cast<int>(stride), 1};
    int dilationA[2] = {1, 1};
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 2, padA, strideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    if (groups > 1) {
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, static_cast<int>(groups)));
    }

    size_t workspace_size = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        input_desc, input_contig.data_ptr<float>(),
        filter_desc, weight_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace, workspace_size,
        &beta,
        output_desc, output_contig.data_ptr<float>()));

    if (bias.has_value()) {
        CHECK_CUDNN(cudnnAddTensor(handle, &alpha, bias_desc, bias.value().contiguous().data_ptr<float>(), &alpha, output_desc, output_contig.data_ptr<float>()));
    }

    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d", &conv_transpose1d, "ConvTranspose1d via cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("normalize", &normalize, "L2 normalization along dim=1",
          pybind11::arg("input"), pybind11::arg("p"), pybind11::arg("dim"));
    m.def("conv1d", &conv1d, "Conv1d via cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("groups"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
