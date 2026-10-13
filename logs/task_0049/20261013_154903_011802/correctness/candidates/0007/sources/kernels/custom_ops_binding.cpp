#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include <string>

#include "../binding_registry.h"

// Declare the launcher from custom_ops.cu
extern "C" void add_channel_bias_round_launcher(
    float* out,
    const float* x,
    const float* bias,
    int batches,
    int channels,
    int length,
    int out_batch_stride,
    int out_channel_stride,
    int in_batch_stride,
    int in_channel_stride,
    cudaStream_t stream);

// Helper for cuDNN error checking
static void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// cuDNN-based transposed convolution without bias
// Implements F.conv_transpose1d(x, weight, bias=None, stride, padding, output_padding)
// Inputs:
//   x: [N, C_in, L_in] float32 CUDA tensor (supports strides)
//   weight: [C_in, C_out, K] float32 CUDA tensor (supports strides)
//   stride, padding, output_padding: ints
// Output:
//   [N, C_out, L_out] float32
at::Tensor conv_transpose1d_no_bias(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C_in, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0 && output_padding < stride, "output_padding must be in [0, stride-1]");

    // Dimensions
    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_in_w = weight.size(0);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);

    TORCH_CHECK(C_in == C_in_w, "x and weight have mismatched input channels");

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;

    // Ensure contiguous tensors for cuDNN (supports strides but cuDNN requires contiguous or specific strides)
    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();

    // Create output tensor (contiguous, same dtype, CUDA)
    auto y = at::empty({N, C_out, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle and tensor descriptors
    cudnnHandle_t handle;
    checkCudnn(cudnnCreate(&handle), "cudnnCreate failed");
    cudnnSetStream(handle, stream);

    // Tensor descriptor for x: [N, C_in, L_in] -> NCHW format in 2D interpretation (N, C, H, W)
    // For 1D conv, we treat as 2D with height=1 and width=L_in
    cudnnTensorDescriptor_t x_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x_desc failed");
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          N, C_in, 1, L_in), "cudnnSetTensor4dDescriptor x failed");

    // Filter descriptor for weight: [K, C_in, C_out] in cuDNN convention (filter is [output_channels, input_channels, kernel])
    cudnnFilterDescriptor_t w_desc;
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor w_desc failed");
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          C_out, C_in, 1, K), "cudnnSetFilter4dDescriptor w failed");

    // Output descriptor: [N, C_out, L_out] -> [N, C_out, 1, L_out]
    cudnnTensorDescriptor_t y_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y_desc failed");
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          N, C_out, 1, L_out), "cudnnSetTensor4dDescriptor y failed");

    // Convolution descriptor: set as 2D conv with 1xK kernel, but we need transposed conv
    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");
    // For transposed conv, we use cudnnSetConvolution2dDescriptor with conv mode CROSS_CORRELATION (typical)
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                              1, K,               // pad_h, pad_w (pad only width)
                                              stride, 1,          // stride_h, stride_w (stride only width? Actually for transposed conv, stride is the stride of the conv)
                                              1, 1,               // dilation_h, dilation_w
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT),
               "cudnnSetConvolution2dDescriptor failed");

    // We will use cudnnConvolutionBackwardData, which is equivalent to transposed convolution.
    // Need to compute the convolution forward algorithm for data gradient.
    cudnnConvolutionBwdDataAlgo_t algo;
    checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm(handle,
                                                        w_desc,
                                                        y_desc,
                                                        conv_desc,
                                                        x_desc,
                                                        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                        0,
                                                        &algo),
               "cudnnGetConvolutionBackwardDataAlgorithm failed");

    // Workspace size
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                            w_desc,
                                                            y_desc,
                                                            conv_desc,
                                                            x_desc,
                                                            algo,
                                                            &workspace_size),
               "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Execute: dy = y, dw = weight; computes dx = alpha * ConvBackwardData(w, dy) + beta * dx
    // Here x_desc is dx, y_desc is dy, w_desc is w.
    checkCudnn(cudnnConvolutionBackwardData(handle,
                                           &alpha,
                                           w_desc, w_contig.data_ptr<float>(),
                                           y_desc, x_contig.data_ptr<float>(),  // note: y (input to backward) is x_contig (our actual input x)
                                           conv_desc,
                                           algo,
                                           workspace, workspace_size,
                                           &beta,
                                           x_desc, y.data_ptr<float>()),  // x (output of backward) is our output y
               "cudnnConvolutionBackwardData failed");

    if (workspace) {
        cudaFree(workspace);
    }

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return y;
}

// PyTorch wrapper for add_channel_bias_round
at::Tensor add_channel_bias_round(
    const at::Tensor& x,
    const at::Tensor& bias) {
    // Validate
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C]");
    TORCH_CHECK(x.size(1) == bias.size(0), "bias size must match channel dimension");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    auto y = at::empty_like(x);

    // Get strides (in elements)
    int64_t in_batch_stride = x.stride(0);
    int64_t in_channel_stride = x.stride(1);
    int64_t out_batch_stride = y.stride(0);
    int64_t out_channel_stride = y.stride(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_channel_bias_round_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(out_batch_stride),
        static_cast<int>(out_channel_stride),
        static_cast<int>(in_batch_stride),
        static_cast<int>(in_channel_stride),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_no_bias", &conv_transpose1d_no_bias,
          "1D transposed convolution without bias (cuDNN)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));

    m.def("add_channel_bias_round", &add_channel_bias_round,
          "Add channel bias and round",
          pybind11::arg("x"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
