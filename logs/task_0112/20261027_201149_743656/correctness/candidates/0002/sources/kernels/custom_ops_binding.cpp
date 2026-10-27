#include <torch/extension.h>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_dim1_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_l,
    int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_l,
    cudaStream_t stream);

extern "C" void custom_conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

// Helper macro for cuDNN error checking
#define CUDNN_CHECK(expr)                                                        \
    do {                                                                         \
        cudnnStatus_t status = (expr);                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(std::string("cuDNN error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) +          \
                                     " code " + std::to_string(status));        \
        }                                                                        \
    } while (0)

// ConvTranspose1d implementation using cuDNN
torch::Tensor custom_conv_transpose1d_cudnn(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t K = weight.size(2);
    int64_t C_out = weight.size(1);

    TORCH_CHECK(C_in == weight.size(0), "weight size[0] must match input channels");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    cudnnTensorDescriptor_t x_desc, y_desc, w_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));

    int dims = 5;
    int input_n = static_cast<int>(N);
    int input_c = static_cast<int>(C_in);
    int input_d = static_cast<int>(L_in);
    int input_h = 1, input_w = 1;
    int input_stride[5] = {input_c * input_d * input_h * input_w, input_d * input_h * input_w, input_h * input_w, input_w, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, dims, (int[]){input_n, input_c, input_d, input_h, input_w}, input_stride));

    int output_n = static_cast<int>(N);
    int output_c = static_cast<int>(C_out);
    int output_d = static_cast<int>(L_out);
    int output_h = 1, output_w = 1;
    int output_stride[5] = {output_c * output_d * output_h * output_w, output_d * output_h * output_w, output_h * output_w, output_w, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, dims, (int[]){output_n, output_c, output_d, output_h, output_w}, output_stride));

    int filter_dim = 5;
    int filter_dims[5] = {static_cast<int>(C_out), static_cast<int>(C_in), 1, 1, static_cast<int>(K)};
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_dim, filter_dims));

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    int padA[3] = {0, 0, static_cast<int>(padding)};
    int strideA[3] = {1, 1, static_cast<int>(stride)};
    int dilationA[3] = {1, 1, 1};
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, strideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t workspace_size = 0;
    int requested_algo_count = 1;
    cudnnConvolutionBwdDataAlgoPerf_t perf;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc, conv_desc, x_desc, 1, &requested_algo_count, &perf));
    algo = perf.algo;
    workspace_size = perf.memory;

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc workspace failed");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        w_desc, weight_contig.data_ptr<float>(),
        y_desc, input_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        x_desc, output.data_ptr<float>()));

    if (workspace) cudaFree(workspace);

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return output;
}

// Softmax dim=1 wrapper
torch::Tensor custom_softmax_dim1(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C, L]");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto output = torch::empty_like(input);

    int64_t in_stride_b = input.stride(0);
    int64_t in_stride_c = input.stride(1);
    int64_t in_stride_l = input.stride(2);
    int64_t out_stride_b = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_l = output.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_dim1_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(C), static_cast<int>(L),
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l,
        stream);

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d_cudnn,
          "Custom ConvTranspose1d using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("stride") = 2,
          py::arg("padding") = 1, py::arg("output_padding") = 1);
    m.def("custom_softmax_dim1", &custom_softmax_dim1,
          "Custom softmax over dim=1",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
