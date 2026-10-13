#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void leaky_relu_launcher(float* output, const float* input, float negative_slope, long long total_elements, cudaStream_t stream);
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias, int C, int HW, long long total_elements, cudaStream_t stream);

// Helper to get cuDNN handle for the current device and stream
cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);
    return handle;
}

// cuDNN convolution forward with bias
// input: [N, C, H, W], weight: [K, C, R, S], bias: [K]
// stride, padding, dilation, groups are integers
// returns output [N, K, H_out, W_out]
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    // Parameter validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K, C/groups, R, S)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(groups > 0, "groups must be positive");

    // Make contiguous if necessary; cuDNN requires contiguous tensors
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);
    int K = weight_contig.size(0);
    int R = weight_contig.size(2);
    int S = weight_contig.size(3);

    TORCH_CHECK(K == bias_contig.size(0), "bias size must match output channels");
    TORCH_CHECK(weight_contig.size(1) == C / groups, "weight has incorrect input channels per group");
    TORCH_CHECK(C % groups == 0, "input channels must be divisible by groups");

    // Compute output dimensions
    int H_out = (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (S - 1) - 1) / stride + 1;

    auto output = torch::empty({N, K, H_out, W_out}, input_contig.options());

    // Get cuDNN handle
    cudnnHandle_t handle = get_cudnn_handle();

    // Set up tensor descriptors
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, H_out, W_out);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C / groups, R, S);

    // cuDNN convolution descriptor: supports groups
    cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding, padding,
        stride, stride,
        dilation, dilation,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );
    if (groups > 1) {
        cudnnSetConvolutionGroupCount(conv_desc, groups);
    }

    // Determine algorithm (choose a deterministic heuristic)
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        1,  // requested algo count
        nullptr,
        &algo
    );

    // Workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size
    );

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        handle,
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
        output.data_ptr<float>()
    );

    // Add bias: use cuDNN's add tensor with broadcast to NCHW
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    // bias shape [K, 1, 1] in NCHW, but cuDNN AddTensor expects same dimensions; we can use bias with dimensions [1, K, 1, 1]
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1);
    float alpha_bias = 1.0f, beta_bias = 1.0f;
    cudnnAddTensor(
        handle,
        &alpha_bias,
        bias_desc,
        bias_contig.data_ptr<float>(),
        &beta_bias,
        output_desc,
        output.data_ptr<float>()
    );

    // Cleanup cuDNN descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    if (workspace) cudaFree(workspace);

    // Check for cuDNN errors
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in conv2d_forward: ", cudaGetErrorString(cuda_err));

    return output;
}

// leaky_relu_forward
torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<float>(negative_slope),
        input.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in leaky_relu_forward: ", cudaGetErrorString(err));
    return output;
}

// add_bias_forward
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D [C,1,1]");

    int C = input.size(1);
    int HW = input.size(2) * input.size(3);
    TORCH_CHECK(bias.size(0) == C, "bias channel dimension must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape [C,1,1]");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        C,
        HW,
        input.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in add_bias_forward: ", cudaGetErrorString(err));
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "2D convolution forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("groups"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          pybind11::arg("input"),
          pybind11::arg("negative_slope"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias with broadcasting",
          pybind11::arg("input"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
