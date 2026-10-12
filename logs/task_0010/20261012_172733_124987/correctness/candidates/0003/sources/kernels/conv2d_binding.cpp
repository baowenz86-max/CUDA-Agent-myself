#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <stdexcept>
#include <vector>
#include "../binding_registry.h"

#define CHECK_CUDNN(expr)                                                                 \
    do {                                                                                 \
        cudnnStatus_t status = (expr);                                                   \
        if (status != CUDNN_STATUS_SUCCESS) {                                            \
            throw std::runtime_error(std::string("cuDNN error: ") +                     \
                                     cudnnGetErrorString(status));                       \
        }                                                                                \
    } while (0)

extern "C" void conv2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream
);

static cudnnHandle_t cudnn_handle = nullptr;

static void ensure_cudnn_handle() {
    if (!cudnn_handle) {
        CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    }
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC,IC,KH,KW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels, "weight channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias size mismatch");
    TORCH_CHECK(kernel_h % 2 == 1 && kernel_w % 2 == 1, "kernel must be odd");
    int out_h = (in_h + 2 * (kernel_h / 2) - kernel_h) + 1;
    int out_w = (in_w + 2 * (kernel_w / 2) - kernel_w) + 1;

    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    ensure_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filt_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, in_channels, in_h, in_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, out_channels, out_h, out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_h, kernel_w));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, kernel_h / 2, kernel_w / 2, 1, 1, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, in_desc, filt_desc, conv_desc,
                                                    out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0, &algo));

    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, in_desc, filt_desc, conv_desc,
                                                        out_desc, algo, &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, in_desc, input.data_ptr<float>(),
                                        filt_desc, weight.data_ptr<float>(), conv_desc, algo,
                                        workspace, workspace_size, &beta, out_desc, output.data_ptr<float>()));

    // Add bias using cuDNN's tensor add
    cudnnTensorDescriptor_t bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, out_channels, 1, 1));
    CHECK_CUDNN(cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias.data_ptr<float>(),
                               &alpha, out_desc, output.data_ptr<float>()));

    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filt_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));

    return output;
}

void register_conv2d_forward(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward using cuDNN");
}

REGISTER_BINDING(conv2d_forward, register_conv2d_forward);
