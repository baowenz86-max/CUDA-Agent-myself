// PyTorch binding for custom ops using cuDNN for Conv2d and CUDA kernels for others
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int in_N_stride, int in_C_stride, int in_H_stride, int in_W_stride,
    int out_N_stride, int out_C_stride, int out_H_stride, int out_W_stride,
    cudaStream_t stream);

extern "C" void batch_norm_launcher(
    const float* input,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    float* output,
    float eps,
    int N, int C, int H, int W,
    int in_N_stride, int in_C_stride, int in_H_stride, int in_W_stride,
    int out_N_stride, int out_C_stride, int out_H_stride, int out_W_stride,
    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    int in_N_stride, int in_C_stride, int in_H_stride, int in_W_stride,
    int out_N_stride, int out_C_stride, int out_H_stride, int out_W_stride,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int64_t total,
    int64_t in_stride,
    int64_t out_stride,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Conv2d using cuDNN
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                             int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    // Convert stride/padding to tuple if needed (they are ints)
    int stride_h = stride;
    int stride_w = stride;
    int padding_h = padding;
    int padding_w = padding;

    // Ensure contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.dim() == 1,
                    "Bias must be 1D float32 CUDA tensor");
        bias_contig = bias.contiguous();
    }

    int N = input_contig.size(0);
    int C_in = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);
    int K = weight_contig.size(0);
    int C_out = K;
    int R = weight_contig.size(2);
    int S = weight_contig.size(3);

    TORCH_CHECK(weight_contig.size(1) == C_in, "Weight channel mismatch");

    int H_out = (H + 2 * padding_h - R) / stride_h + 1;
    int W_out = (W + 2 * padding_w - S) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, getCurrentStream());

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C_in, R, S);
    cudnnSetConvolution2dDescriptor(conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input_contig.data_ptr<float>(),
                            filter_desc, weight_contig.data_ptr<float>(), conv_desc,
                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta,
                            output_desc, output.data_ptr<float>());

    if (bias_contig.defined()) {
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        cudnnAddTensor(cudnn_handle, &alpha_bias, bias_desc, bias_contig.data_ptr<float>(),
                       &beta_bias, output_desc, output.data_ptr<float>());
    }

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// Add bias forward
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1, "Bias shape must be [C,1,1]");
    TORCH_CHECK(bias.size(0) == input.size(1), "Bias channel mismatch");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    // Since both contiguous, strides can be computed but pass as is
    int in_N_stride = input_contig.stride(0);
    int in_C_stride = input_contig.stride(1);
    int in_H_stride = input_contig.stride(2);
    int in_W_stride = input_contig.stride(3);
    int out_N_stride = output.stride(0);
    int out_C_stride = output.stride(1);
    int out_H_stride = output.stride(2);
    int out_W_stride = output.stride(3);

    add_bias_launcher(input_contig.data_ptr<float>(), bias.data_ptr<float>(),
                      output.data_ptr<float>(), N, C, H, W,
                      in_N_stride, in_C_stride, in_H_stride, in_W_stride,
                      out_N_stride, out_C_stride, out_H_stride, out_W_stride,
                      getCurrentStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias_launcher failed");
    return output;
}

// BatchNorm forward
torch::Tensor batch_norm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    int in_N_stride = input_contig.stride(0);
    int in_C_stride = input_contig.stride(1);
    int in_H_stride = input_contig.stride(2);
    int in_W_stride = input_contig.stride(3);
    int out_N_stride = output.stride(0);
    int out_C_stride = output.stride(1);
    int out_H_stride = output.stride(2);
    int out_W_stride = output.stride(3);

    batch_norm_launcher(input_contig.data_ptr<float>(), running_mean.data_ptr<float>(),
                        running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                        output.data_ptr<float>(), static_cast<float>(eps), N, C, H, W,
                        in_N_stride, in_C_stride, in_H_stride, in_W_stride,
                        out_N_stride, out_C_stride, out_H_stride, out_W_stride,
                        getCurrentStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "batch_norm_launcher failed");
    return output;
}

// AvgPool2d forward
torch::Tensor avg_pool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");

    auto input_contig = input.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid pooling parameters");

    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());

    int in_N_stride = input_contig.stride(0);
    int in_C_stride = input_contig.stride(1);
    int in_H_stride = input_contig.stride(2);
    int in_W_stride = input_contig.stride(3);
    int out_N_stride = output.stride(0);
    int out_C_stride = output.stride(1);
    int out_H_stride = output.stride(2);
    int out_W_stride = output.stride(3);

    avg_pool2d_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(),
                        N, C, H, W, H_out, W_out,
                        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
                        in_N_stride, in_C_stride, in_H_stride, in_W_stride,
                        out_N_stride, out_C_stride, out_H_stride, out_W_stride,
                        getCurrentStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "avg_pool2d_launcher failed");
    return output;
}

// Sigmoid forward
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int64_t total = input_contig.numel();
    if (total == 0) return output;

    sigmoid_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), total, 1, 1, getCurrentStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sigmoid_launcher failed");
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias forward",
          py::arg("input"), py::arg("bias"));
    m.def("batch_norm_forward", &batch_norm_forward, "BatchNorm forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("running_mean"),
          py::arg("running_var"), py::arg("eps"));
    m.def("avg_pool2d_forward", &avg_pool2d_forward, "AvgPool2d forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
