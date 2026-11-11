#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <iostream>
#include "../binding_registry.h"

// Forward declaration of launcher from custom_ops.cu
extern "C" void logsumexp_dim_forward_launcher(
    const float* input,
    float* output,
    int B,
    int C,
    int L,
    int input_stride_b,
    int input_stride_c,
    bool keepdim,
    int output_stride_b,
    int output_stride_c,
    cudaStream_t stream
);

// Helper to check cuDNN status
#define CHECK_CUDNN(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        } \
    } while (0)

// ConvTranspose1d implementation using cuDNN
static torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(1), "bias shape mismatch");
    }

    // Ensure contiguous for cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    // Allocate output
    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    // cuDNN handle and tensor descriptors
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    cudnnTensorDescriptor_t input_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    // Input tensor: [N, C_in, L_in] -> cuDNN expects NCHW, use 1D as width
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C_in, L_in, 1
    ));

    // Output tensor: [N, C_out, L_out]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C_out, L_out, 1
    ));

    // Filter descriptor: ConvTranspose uses shape [C_in, C_out, 1, kernel_size]
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        C_in, C_out, 1, kernel_size
    ));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding, 0,          // pad_h, pad_w (only width padding)
        stride, 1,            // stride_h, stride_w
        1, 1,                 // dilation_h, dilation_w
        CUDNN_CROSS_CORRELATION, // mode (for transpose conv, cuDNN uses CROSS_CORRELATION? Actually use CUDNN_CONVOLUTION)
        CUDNN_DATA_FLOAT
    ));

    // For conv transpose, we need to use cudnnConvolutionBackwardData or cudnnConvolutionForward with CUDNN_CONVOLUTION_BWD_DATA? 
    // Better: use cudnnConvolutionBackwardData which computes the transposed convolution.
    // Reference: cudnnConvolutionBackwardData(cudnnHandle_t, const void* alpha, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx)
    // Here dy is input (gradient wrt output), dx is output (gradient wrt input).

    float alpha = 1.0f, beta = 0.0f;

    // Select algorithm (optional; can use default heuristic)
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        &beta,
        output_desc,
        output.data_ptr<float>()
    ));

    // Add bias if provided
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        auto bias_contig = bias.contiguous();
        // Use cudnnAddTensor to add bias per channel: shape [1, C_out, 1, 1] tensor descriptor
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            bias_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, C_out, 1, 1
        ));
        float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(
            cudnn_handle,
            &alpha,
            bias_desc,
            bias_contig.data_ptr<float>(),
            &beta_bias,
            output_desc,
            output.data_ptr<float>()
        ));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    // Check for CUDA errors after cuDNN calls
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after conv_transpose1d: ") + cudaGetErrorString(cuda_err));
    }

    return output;
}

// Logsumexp along dim (currently only supports dim == input.dim()-1, else generic fallback)
static torch::Tensor logsumexp_dim_forward(
    const torch::Tensor& input,
    int64_t dim,
    bool keepdim
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(dim >= -input.dim() && dim < input.dim(), "dim out of range");

    // Normalize dim
    int64_t dim_norm = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim_norm == input.dim() - 1, "only last dim logsumexp is implemented");

    // For now we assume 3D input [B, C, L] because our model uses that after conv transpose.
    TORCH_CHECK(input.dim() == 3, "expected 3D tensor [B, C, L] for logsumexp");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    // Get strides (after contiguous? Our kernel uses strides, but input could be non-contiguous due to previous operation.
    // To be safe, make contiguous.
    auto input_contig = input.contiguous();

    // Allocate output shape
    std::vector<int64_t> output_sizes;
    if (keepdim) {
        output_sizes = {B, C, 1};
    } else {
        output_sizes = {B, C};
    }
    auto output = torch::empty(output_sizes, input_contig.options());

    // Compute strides for output
    int64_t output_stride_b, output_stride_c;
    if (keepdim) {
        output_stride_b = C * 1;
        output_stride_c = 1;
    } else {
        output_stride_b = C;
        output_stride_c = 1;
    }

    // Input strides after contiguous: B stride = C*L, C stride = L
    int64_t input_stride_b = C * L;
    int64_t input_stride_c = L;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    logsumexp_dim_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(input_stride_b),
        static_cast<int>(input_stride_c),
        keepdim,
        static_cast<int>(output_stride_b),
        static_cast<int>(output_stride_c),
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after logsumexp: ") + cudaGetErrorString(err));
    }

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", [](const torch::Tensor& input, const torch::Tensor& weight, const c10::optional<torch::Tensor>& bias, int64_t stride, int64_t padding, int64_t output_padding) {
        return conv_transpose1d_forward(input, weight, bias, stride, padding, output_padding);
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("output_padding") = 0);

    m.def("logsumexp_dim_forward", [](const torch::Tensor& input, int64_t dim, bool keepdim) {
        return logsumexp_dim_forward(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim"), py::arg("keepdim") = false);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
