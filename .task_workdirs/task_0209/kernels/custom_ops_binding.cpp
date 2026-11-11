#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
// Note: exact signature must match; here we declare as it is in the .cu file
extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int batch,
    int channels,
    int length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long in_stride_batch,
    long long in_stride_channel,
    long long t1_stride_batch,
    long long t1_stride_channel,
    long long t2_stride_batch,
    long long t2_stride_channel,
    cudaStream_t stream
);

// Helper to check cuDNN status
#define CHECK_CUDNN(expr)                                                       \
  do {                                                                          \
    cudnnStatus_t status = (expr);                                              \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                                 \
                "cuDNN error: ", cudnnGetErrorString(status));                  \
  } while (0)

// ConvTranspose1d implemented with cuDNN
// input: [batch, in_channels, in_length]
// weight: [in_channels, out_channels, kernel_size]
// bias: [out_channels]
// stride, padding
// output: [batch, out_channels, out_length]
torch::Tensor conv_transpose1d_custom(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias size mismatch");

    // Compute output length
    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size;
    TORCH_CHECK(out_length > 0, "invalid output length");

    // Ensure contiguous for cuDNN (handles arbitrary strides by copying if necessary)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Allocate output tensor
    auto output = torch::empty({batch, out_channels, out_length}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Set tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));

    // For 1D convolution, cuDNN uses 3D tensors with one spatial dimension = 1
    // We use NCHW layout: N=batch, C=channels, H=1, W=length
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch, in_channels, 1, in_length));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch, out_channels, 1, out_length));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, out_channels, 1, 1));

    // Set filter descriptor: for conv_transpose, filter is [in_channels, out_channels, kernel_size]
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           in_channels, out_channels, kernel_size, 1));

    // Convolution descriptor: use cudnnConvolutionBackwardData to implement transposed conv
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    // Set padding and stride for the forward convolution (which is backward data for transpose)
    // For transposed conv, we specify padding and stride of the equivalent forward convolution.
    // Note: output_padding is 0, dilation=1.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, padding,  // vertical, horizontal pad
                                                stride, stride,    // vertical, horizontal stride
                                                1, 1,              // dilation
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Determine algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc,
        input_desc,   // note: in backward data, input_desc is actually the gradient output, but we reuse as input for transpose? Need to check.
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo));

    // Workspace
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        input_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(workspace != nullptr, "failed to allocate cuDNN workspace");
    }

    float alpha = 1.0f, beta = 0.0f;
    // Perform transposed convolution (backward data of a regular convolution)
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>()));

    // Add bias
    CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                               &alpha,
                               bias_desc,
                               bias_contig.data_ptr<float>(),
                               &alpha,
                               output_desc,
                               output.data_ptr<float>()));

    // Cleanup
    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    return output;
}

// addcmul_custom: output = input + tensor1 * tensor2
torch::Tensor addcmul_custom(
    torch::Tensor input,
    torch::Tensor tensor1,
    torch::Tensor tensor2
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(tensor1.is_cuda(), "tensor1 must be a CUDA tensor");
    TORCH_CHECK(tensor2.is_cuda(), "tensor2 must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(tensor1.dtype() == torch::kFloat32, "tensor1 must be float32");
    TORCH_CHECK(tensor2.dtype() == torch::kFloat32, "tensor2 must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(tensor1.dim() == 3, "tensor1 must be 3D");
    TORCH_CHECK(tensor2.dim() == 3, "tensor2 must be 3D");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t length = input.size(2);
    TORCH_CHECK(tensor1.size(0) == batch && tensor1.size(1) == channels && tensor1.size(2) == length,
                "tensor1 shape mismatch");
    TORCH_CHECK(tensor2.size(0) == batch && tensor2.size(1) == channels && tensor2.size(2) == length,
                "tensor2 shape mismatch");

    // Ensure contiguous for simplicity; kernel supports strides but we can pass actual strides
    // However, our kernel handles strides via parameters, so we pass them directly.
    // But for safety, we can call .contiguous() and then use default strides. Since tensors are likely contiguous,
    // this adds negligible overhead. We'll use contiguous to simplify.
    auto input_contig = input.contiguous();
    auto tensor1_contig = tensor1.contiguous();
    auto tensor2_contig = tensor2.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute strides in elements
    long long out_stride_batch = output.stride(0);
    long long out_stride_channel = output.stride(1);
    long long in_stride_batch = input_contig.stride(0);
    long long in_stride_channel = input_contig.stride(1);
    long long t1_stride_batch = tensor1_contig.stride(0);
    long long t1_stride_channel = tensor1_contig.stride(1);
    long long t2_stride_batch = tensor2_contig.stride(0);
    long long t2_stride_channel = tensor2_contig.stride(1);

    addcmul_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        tensor1_contig.data_ptr<float>(),
        tensor2_contig.data_ptr<float>(),
        (int)batch,
        (int)channels,
        (int)length,
        out_stride_batch,
        out_stride_channel,
        in_stride_batch,
        in_stride_channel,
        t1_stride_batch,
        t1_stride_channel,
        t2_stride_batch,
        t2_stride_channel,
        stream
    );

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_custom", &conv_transpose1d_custom,
          "ConvTranspose1d using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));
    m.def("addcmul_custom", &addcmul_custom,
          "addcmul: input + tensor1 * tensor2",
          py::arg("input"), py::arg("tensor1"), py::arg("tensor2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
