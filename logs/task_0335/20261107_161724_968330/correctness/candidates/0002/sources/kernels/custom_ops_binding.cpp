#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "../binding_registry.h"

// Launcher declarations
extern "C" void linear_forward_launcher(float* output,
                                        const float* input,
                                        const float* weight,
                                        const float* bias,
                                        int batch,
                                        int in_features,
                                        int out_features,
                                        cudaStream_t stream);

extern "C" void interpolate_nearest_1d_launcher(float* output,
                                                 const float* input,
                                                 int batch,
                                                 int channels,
                                                 int length_in,
                                                 int length_out,
                                                 float scale_factor,
                                                 cudaStream_t stream);

static void check_cudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// ------------------- linear_forward -------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "linear_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "linear_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "linear_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 2, "linear_forward: input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "linear_forward: weight must be 2D");
    TORCH_CHECK(input.size(1) == weight.size(1), "linear_forward: input feature dim mismatch");
    TORCH_CHECK(input.is_contiguous(), "linear_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "linear_forward: weight must be contiguous");

    int batch = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "linear_forward: bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "linear_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "linear_forward: bias shape mismatch");
        TORCH_CHECK(bias.is_contiguous(), "linear_forward: bias must be contiguous");
    }

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    linear_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        batch,
        in_features,
        out_features,
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("linear_forward kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}

// ------------------- unsqueeze_forward -------------------
torch::Tensor unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "unsqueeze_forward: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "unsqueeze_forward: input must be float32");
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim + 1;
    TORCH_CHECK(dim >= 0 && dim <= ndim, "unsqueeze_forward: invalid dim");
    return input.unsqueeze(dim);
}

// ------------------- interpolate_nearest_1d -------------------
torch::Tensor interpolate_nearest_1d(torch::Tensor input, double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "interpolate_nearest_1d: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "interpolate_nearest_1d: input must be float32");
    TORCH_CHECK(input.dim() == 3, "interpolate_nearest_1d: input must be [batch, channels, length]");
    TORCH_CHECK(input.is_contiguous(), "interpolate_nearest_1d: input must be contiguous");

    int batch = input.size(0);
    int channels = input.size(1);
    int length_in = input.size(2);
    int length_out = static_cast<int>(std::floor(length_in * scale_factor));
    TORCH_CHECK(length_out > 0, "interpolate_nearest_1d: output length must be positive");

    auto output = torch::empty({batch, channels, length_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    interpolate_nearest_1d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        channels,
        length_in,
        length_out,
        static_cast<float>(scale_factor),
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("interpolate_nearest_1d kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}

// ------------------- conv_transpose1d_forward (cuDNN) -------------------
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose1d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be [batch, in_channels, length_in]");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be [in_channels, out_channels, kernel_size]");
    TORCH_CHECK(input.is_contiguous(), "conv_transpose1d_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int length_in = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "conv_transpose1d_forward: weight in_channels mismatch");

    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "conv_transpose1d_forward: bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels, "conv_transpose1d_forward: bias shape mismatch");
        TORCH_CHECK(bias.is_contiguous(), "conv_transpose1d_forward: bias must be contiguous");
    }

    int length_out = (length_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kernel_size + static_cast<int>(output_padding);
    TORCH_CHECK(length_out > 0, "conv_transpose1d_forward: computed output length must be positive");

    auto output = torch::empty({batch, out_channels, length_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t cudnn_handle;
    check_cudnn(cudnnCreate(&cudnn_handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream");

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor(input)");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor(output)");
    check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor(bias)");
    check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    // Descriptors for 4D NCHW with dummy height dimension
    check_cudnn(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           in_channels,
                                           length_in,
                                           1), "cudnnSetTensor4dDescriptor(input)");

    check_cudnn(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           out_channels,
                                           length_out,
                                           1), "cudnnSetTensor4dDescriptor(output)");

    // Filter: PyTorch weight is [in_channels, out_channels, kernel_size]
    // For transposed conv via backward data, filter is [out_channels, in_channels, kernel_size]
    // but we can use the same weight tensor with appropriate descriptor:
    // cuDNN filter descriptor expects [K, C, R, S] = [out_channels, in_channels, kernel_size, 1]
    // but our weight is [in_channels, out_channels, kernel_size].
    // However, cuDNN's backward data algorithm can use a filter descriptor with dimension order [K, C, R, S]
    // and interpret the provided pointer accordingly. Since our weight is [in_channels, out_channels, kernel_size],
    // we need to set the descriptor as [in_channels, out_channels, kernel_size, 1] and use it with
    // cudnnConvolutionBackwardData? That would be wrong because the roles of C and K are swapped.
    // Actually, the transposed convolution can be seen as the gradient of a convolution with
    // input channels = out_channels, output channels = in_channels, filter = weight transposed.
    // cuDNN's backward data computes dx given dy and filter w, with w having shape [K, C, R, S].
    // In our case, dy = input [batch, in_channels, length_in], and we want dx = output [batch, out_channels, length_out].
    // So K should be in_channels and C should be out_channels. But our weight is [in_channels, out_channels, kernel_size],
    // which matches [K, C, R] if we interpret the first dimension as K (output channels of forward conv)
    // and second as C (input channels of forward conv). Then K=in_channels, C=out_channels.
    // That is exactly the weight layout for a forward conv with input channels=out_channels, output channels=in_channels.
    // We can use cudnnConvolutionBackwardData with dy=input, w=weight (as [K=in_channels, C=out_channels, R=kernel_size]),
    // and conv_desc describing a forward convolution with stride, padding, dilation.
    // Then dx will have shape [batch, out_channels, length_out_base] where length_out_base = (length_in-1)*stride - 2*padding + kernel_size.
    // After that we add output_padding zeros at the end.
    // So the filter descriptor should be set as [in_channels, out_channels, kernel_size, 1].
    check_cudnn(cudnnSetFilter4dDescriptor(filter_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           in_channels,   // K
                                           out_channels,  // C
                                           kernel_size,
                                           1), "cudnnSetFilter4dDescriptor");

    int conv_stride[2] = {1, static_cast<int>(stride)};
    int conv_pad[2] = {0, static_cast<int>(padding)};
    int conv_dilation[2] = {1, 1};
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                conv_pad[0], conv_pad[1],
                                                conv_stride[0], conv_stride[1],
                                                conv_dilation[0], conv_dilation[1],
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");

    int base_length_out = length_out - static_cast<int>(output_padding);
    TORCH_CHECK(base_length_out > 0, "base output length must be positive");

    auto tmp_output = torch::empty({batch, out_channels, base_length_out, 1}, input.options());

    cudnnTensorDescriptor_t tmp_output_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&tmp_output_desc), "cudnnCreateTensorDescriptor(tmp_output)");
    check_cudnn(cudnnSetTensor4dDescriptor(tmp_output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           out_channels,
                                           base_length_out,
                                           1), "cudnnSetTensor4dDescriptor(tmp_output)");

    cudnnConvolutionBwdDataAlgo_t algo;
    int requested_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[1];
    check_cudnn(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle,
                                                          filter_desc,
                                                          input_desc,
                                                          conv_desc,
                                                          tmp_output_desc,
                                                          1,
                                                          &requested_algo_count,
                                                          perf_results), "cudnnFindConvolutionBackwardDataAlgorithm");
    algo = perf_results[0].algo;

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(cudnn_handle,
                                             &alpha,
                                             filter_desc,
                                             weight.data_ptr<float>(),
                                             input_desc,
                                             input.data_ptr<float>(),
                                             conv_desc,
                                             algo,
                                             nullptr,
                                             0,
                                             &beta,
                                             tmp_output_desc,
                                             tmp_output.data_ptr<float>()), "cudnnConvolutionBackwardData");

    // Add bias to tmp_output (before padding)
    if (bias.defined()) {
        check_cudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1,
                                               out_channels,
                                               1,
                                               1), "cudnnSetTensor4dDescriptor(bias)");
        float alpha_bias = 1.0f, beta_bias = 1.0f; // accumulate
        check_cudnn(cudnnAddTensor(cudnn_handle,
                                   &alpha_bias,
                                   bias_desc,
                                   bias.data_ptr<float>(),
                                   &beta_bias,
                                   tmp_output_desc,
                                   tmp_output.data_ptr<float>()), "cudnnAddTensor");
    }

    // Copy tmp_output to final output with zero padding at the end
    output.zero_();
    cudaMemcpy2DAsync(output.data_ptr<float>(), length_out * sizeof(float),
                      tmp_output.data_ptr<float>(), base_length_out * sizeof(float),
                      base_length_out * sizeof(float), batch * out_channels,
                      cudaMemcpyDeviceToDevice, stream);

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(tmp_output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("unsqueeze_forward", &unsqueeze_forward,
          "Unsqueeze forward",
          py::arg("input"), py::arg("dim"));
    m.def("interpolate_nearest_1d", &interpolate_nearest_1d,
          "Nearest 1D interpolation",
          py::arg("input"), py::arg("scale_factor"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
