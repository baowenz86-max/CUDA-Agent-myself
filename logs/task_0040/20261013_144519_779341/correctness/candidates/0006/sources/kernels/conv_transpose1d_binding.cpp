#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Helper to check cuDNN status
#define CUDNN_CHECK(condition) \
    do { \
        cudnnStatus_t status = (condition); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        } \
    } while(0)

// Forward declaration of a function to get cudnn handle from cache (simplified: create new handle each call)
// In production, use a per-device handle cache, but for simplicity we create and destroy handle per call.
cudnnHandle_t get_cudnn_handle() {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    return handle;
}

void destroy_cudnn_handle(cudnnHandle_t handle) {
    cudnnDestroy(handle);
}

// Compute output length for ConvTranspose1d
int64_t compute_output_length(int64_t input_length, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding) {
    return (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// cuDNN implementation of ConvTranspose1d
// Input: x [batch, in_channels, length]
// Weight: [in_channels, out_channels, kernel_size]
// Bias: [out_channels] (may be None)
// Output: [batch, out_channels, Lout]
torch::Tensor conv_transpose1d_cudnn(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D [in_channels, out_channels, kernel_size]");
    TORCH_CHECK(input.size(1) == weight.size(0), "Input channels must match weight in_channels");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0 && output_padding < stride, "output_padding must be in [0, stride)");

    int64_t Lout = compute_output_length(length, kernel_size, stride, padding, output_padding);
    TORCH_CHECK(Lout > 0, "Output length must be positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({batch, out_channels, Lout}, input.options());

    // Handle cuDNN
    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // cuDNN descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));

    // Set tensor descriptors: cuDNN uses NCHW for 2D, for 1D we can use NCHW with width dimension 1, or use NCDHW? Better: use NCHW with H=1 and W=length.
    // Actually cuDNN supports 1D convolutions via NCHW with height=1. We'll use input shape [batch, in_channels, 1, length].
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, in_channels, 1, length));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_channels, 1, Lout));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));

    // Filter descriptor: shape [out_channels, in_channels, height(1), kernel_size]
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, 1, kernel_size));

    // Create convolution descriptor for backward data (since transposed conv is backward data of a regular conv)
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 padding, 0,  // padding height, width
                                                 stride, 1,   // stride height, width
                                                 1, 1,        // dilation height, width
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT));

    // Determine algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle,
                                                          filter_desc,
                                                          output_desc,
                                                          conv_desc,
                                                          input_desc,
                                                          CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                          0,
                                                          &algo));

    // Work space
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                              filter_desc,
                                                              output_desc,
                                                              conv_desc,
                                                              input_desc,
                                                              algo,
                                                              &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Execute backward data: input is destination (output of transposed conv), output_desc is the source (input to transposed conv?)
    // For transposed conv (deconvolution), we typically use cudnnConvolutionBackwardData with the roles swapped:
    // dy (gradient wrt output) -> input to transposed conv,
    // dx (gradient wrt input) -> output of transposed conv.
    // Let's define: dy = input tensor (shape [batch, in_channels, length])
    //               dx = output tensor (shape [batch, out_channels, Lout])
    // Then we need to compute dx = transposed_conv(dy, weight) + bias.
    // The cuDNN call is:
    // cudnnConvolutionBackwardData(handle, alpha, filter_desc, filter, dy_desc, dy, conv_desc, algo, workspace, workspace_size, beta, dx_desc, dx);
    // But we need to set the descriptors correctly: dy is the "input" to the backward data, which is the smaller one (input to transposed conv).
    // In our earlier set, input_desc corresponds to dy, and output_desc to dx. So dy_desc = input_desc, dx_desc = output_desc.
    // The filter is the same weight.
    // alpha = 1.0, beta = 0.0 initially, then add bias if present.

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle,
                                              &alpha,
                                              filter_desc, weight.data_ptr<float>(),
                                              input_desc, input.data_ptr<float>(),
                                              conv_desc,
                                              algo,
                                              workspace,
                                              workspace_size,
                                              &beta,
                                              output_desc, output.data_ptr<float>()));

    // Add bias if provided
    if (bias.defined()) {
        // Use cudnnAddTensor (or cudnnAddTensor_v2)
        alpha = 1.0f; beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle,
                                   &alpha,
                                   bias_desc, bias.data_ptr<float>(),
                                   &beta,
                                   output_desc, output.data_ptr<float>()));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    destroy_cudnn_handle(handle);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after conv_transpose1d: ") + cudaGetErrorString(err));
    }

    return output;
}

void register_conv_transpose1d(pybind11::module& m) {
    m.def("conv_transpose1d_cudnn", &conv_transpose1d_cudnn,
          "ConvTranspose1d using cuDNN",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"));
}

REGISTER_BINDING(conv_transpose1d, register_conv_transpose1d);
