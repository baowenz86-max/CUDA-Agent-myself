#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_bias_forward_launcher(float* output, const float* input, const float* bias,
                                          int batch_size, int channels, int length, cudaStream_t stream);
extern "C" void softmax_channel_forward_launcher(float* output, const float* input,
                                                 int batch_size, int channels, int length, cudaStream_t stream);
extern "C" void scale_forward_launcher(float* output, const float* input, float scalar,
                                       int total, cudaStream_t stream);

namespace {

std::string cudnnErrorToString(cudnnStatus_t status) {
    return cudnnGetErrorString(status);
}

void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnErrorToString(status));
    }
}

void checkCuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

torch::Tensor conv_transpose_op_impl(torch::Tensor input, torch::Tensor weight,
                                     torch::Tensor conv_bias, int64_t stride,
                                     int64_t padding, int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (conv_bias.defined() && conv_bias.numel() > 0) {
        TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (conv_bias.defined() && conv_bias.numel() > 0) {
        TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (conv_bias.defined() && conv_bias.numel() > 0) {
        TORCH_CHECK(conv_bias.dtype() == torch::kFloat32, "conv_bias must be float32");
    }

    int64_t B = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t kernel_size = weight.size(2);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    TORCH_CHECK(weight.size(0) == C_in, "weight shape mismatch");
    TORCH_CHECK(B > 0 && C_in > 0 && L_in > 0, "invalid input dimensions");
    TORCH_CHECK(L_out > 0, "invalid output length");

    auto output = torch::empty({B, C_out, L_out}, input.options());

    cudnnHandle_t handle;
    checkCudnn(cudnnCreate(&handle), "cudnnCreate failed");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    checkCudnn(cudnnSetStream(handle, stream), "cudnnSetStream failed");

    // Set up tensor descriptors for NCHW layout, but treat L as W dimension.
    cudnnTensorDescriptor_t x_desc, y_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x failed");
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y failed");

    // For ConvTranspose1d, PyTorch uses NCL layout; cuDNN expects NCHW, so set H=1.
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           B, C_in, 1, L_in), "cudnnSetTensor4dDescriptor x failed");
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           B, C_out, 1, L_out), "cudnnSetTensor4dDescriptor y failed");

    cudnnFilterDescriptor_t w_desc;
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor failed");
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           C_in, C_out, 1, kernel_size), "cudnnSetFilter4dDescriptor failed");

    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padding, 0,  // vertical pad, horizontal pad=0 since H=1
                                               stride, 1,    // vertical stride, horizontal stride=1
                                               1, 1,          // dilation
                                               CUDNN_CROSS_CORRELATION,  // PyTorch uses cross-correlation for conv_transpose
                                               CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor failed");

    // For transposed convolution, use cudnnConvolutionBackwardData.
    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnConvolutionBackwardData(handle,
                                            &alpha,
                                            w_desc, weight.data_ptr<float>(),
                                            x_desc, input.data_ptr<float>(),
                                            conv_desc,
                                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,  // use algo 1 for simplicity; can auto-tune later
                                            nullptr, 0,
                                            &beta,
                                            y_desc, output.data_ptr<float>()),
               "cudnnConvolutionBackwardData failed");

    if (conv_bias.defined() && conv_bias.numel() > 0) {
        // Add conv bias: y += bias_c for each channel
        float bias_alpha = 1.0f, bias_beta = 1.0f;
        cudnnTensorDescriptor_t bias_desc;
        checkCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
        checkCudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, C_out, 1, 1), "cudnnSetTensor4dDescriptor bias failed");
        checkCudnn(cudnnAddTensor(handle, &bias_alpha, bias_desc, conv_bias.data_ptr<float>(),
                                  &bias_beta, y_desc, output.data_ptr<float>()),
                   "cudnnAddTensor bias failed");
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return output;
}

torch::Tensor add_bias_op_impl(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B,C,L]");
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == input.size(1) && bias.size(1) == 1,
                "bias must be shape [C,1] matching input channels");
    auto output = torch::empty_like(input);
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                              bias.data_ptr<float>(), B, C, L, stream);
    checkCuda(cudaGetLastError(), "add_bias_forward_launcher failed");
    return output;
}

torch::Tensor softmax_channel_op_impl(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B,C,L]");
    auto output = torch::empty_like(input);
    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_channel_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                                     B, C, L, stream);
    checkCuda(cudaGetLastError(), "softmax_channel_forward_launcher failed");
    return output;
}

torch::Tensor scale_op_impl(torch::Tensor input, double scaling_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input);
    int64_t total = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    scale_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                           static_cast<float>(scaling_factor), total, stream);
    checkCuda(cudaGetLastError(), "scale_forward_launcher failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose_op", &conv_transpose_op_impl,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_op", &add_bias_op_impl,
          "Add bias with shape [C,1] broadcast to [B,C,L]",
          py::arg("input"), py::arg("bias"));
    m.def("softmax_channel_op", &softmax_channel_op_impl,
          "Softmax over channel dimension dim=1",
          py::arg("input"));
    m.def("scale_op", &scale_op_impl,
          "Multiply tensor by scalar",
          py::arg("input"), py::arg("scaling_factor"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
