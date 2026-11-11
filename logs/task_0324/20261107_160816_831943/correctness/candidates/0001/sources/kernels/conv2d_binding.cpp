#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// PyTorch wrapper for Conv2d using cuDNN
torch::Tensor conv2d(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt, int64_t stride, int64_t padding) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_out, C_in, kH, kW)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(0);
    int C_in_w = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    TORCH_CHECK(C_in == C_in_w, "weight input channels mismatch");

    int H_out = (H_in - kH + 2 * static_cast<int>(padding)) / static_cast<int>(stride) + 1;
    int W_out = (W_in - kW + 2 * static_cast<int>(padding)) / static_cast<int>(stride) + 1;

    // Allocate output
    auto y = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Ensure input and weight are contiguous for cuDNN
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    // Get cuDNN handle
    cudnnHandle_t handle;
    TORCH_CHECK(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS, "cuDNN create failed");

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    TORCH_CHECK(cudnnCreateTensorDescriptor(&x_desc) == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor failed");
    TORCH_CHECK(cudnnCreateTensorDescriptor(&y_desc) == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor failed");
    TORCH_CHECK(cudnnCreateFilterDescriptor(&w_desc) == CUDNN_STATUS_SUCCESS, "cudnnCreateFilterDescriptor failed");
    TORCH_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc) == CUDNN_STATUS_SUCCESS, "cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors (NCHW format)
    TORCH_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H_in, W_in) == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor x failed");
    TORCH_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out) == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor y failed");

    // Set filter descriptor: format CUDNN_TENSOR_NCHW means filter is [C_out, C_in, kH, kW]
    TORCH_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kH, kW) == CUDNN_STATUS_SUCCESS, "cudnnSetFilter4dDescriptor failed");

    // Set convolution descriptor
    TORCH_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                static_cast<int>(padding), static_cast<int>(padding),
                                                static_cast<int>(stride), static_cast<int>(stride),
                                                1, 1, // dilation
                                                CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
                                                CUDNN_DATA_FLOAT) == CUDNN_STATUS_SUCCESS,
                "cudnnSetConvolution2dDescriptor failed");

    // Determine algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    // Allocate workspace
    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size);
    void* workspace = nullptr;
    if (ws_size > 0) {
        cudaMalloc(&workspace, ws_size);
    }

    // Get stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    float alpha = 1.0f, beta = 0.0f;
    if (bias_opt.has_value()) {
        // With bias, we need to add bias after convolution. PyTorch Conv2d with bias adds bias to each output channel.
        // We can set beta=1.0 and use a bias tensor in the output, but cuDNN convolution does not include bias directly.
        // Instead, we can use cudnnConvolutionBiasActivationForward or simply do convolution with beta=0 and add bias separately.
        // For simplicity, we do convolution without bias and then add bias with a custom kernel? But we can't use torch ops in C++.
        // However, we can use cudnnAddTensor or use a simple custom kernel? But task says .cpp cannot contain CUDA kernels.
        // We can use cuDNN's ConvolutionBiasActivationForward or use cudnnAddTensor.
        // Here we use cudnnAddTensor to add bias.
        auto bias_contig = bias_opt.value().contiguous();
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        // Bias shape [C_out] should be treated as [1, C_out, 1, 1] for NCHW broadcast.
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);

        cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha, x_desc, x_contig.data_ptr<float>(),
                                                       w_desc, weight_contig.data_ptr<float>(),
                                                       conv_desc, algo, workspace, ws_size,
                                                       &beta, y_desc, y.data_ptr<float>());
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward failed");

        // Add bias
        float add_alpha = 1.0f, add_beta = 1.0f;
        status = cudnnAddTensor(handle, &add_alpha, bias_desc, bias_contig.data_ptr<float>(),
                                &add_beta, y_desc, y.data_ptr<float>());
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnAddTensor bias failed");

        cudnnDestroyTensorDescriptor(bias_desc);
    } else {
        cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha, x_desc, x_contig.data_ptr<float>(),
                                                       w_desc, weight_contig.data_ptr<float>(),
                                                       conv_desc, algo, workspace, ws_size,
                                                       &beta, y_desc, y.data_ptr<float>());
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward failed");
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return y;
}

void register_conv2d(pybind11::module& m) {
    m.def("conv2d", &conv2d,
          "Conv2d using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(conv2d, register_conv2d);
