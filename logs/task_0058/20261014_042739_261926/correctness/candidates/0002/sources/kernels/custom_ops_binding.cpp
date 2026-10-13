#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_dim_launcher(float* output, const float* input,
                                 int dim, int batch, int channels, int length,
                                 long long in_stride_batch, long long in_stride_channel, long long in_stride_length,
                                 long long out_stride_batch, long long out_stride_length,
                                 cudaStream_t stream);

extern "C" void conv1d_launcher(const float* input, const float* weight, const float* bias,
                                float* output, int batch, int in_channels, int out_channels, int length,
                                int kernel_size, int stride, int padding, int output_length,
                                cudaStream_t stream);

extern "C" void conv_transpose1d_launcher(const float* input, const float* weight, const float* bias,
                                          float* output, int batch, int in_channels, int out_channels, int length_in,
                                          int kernel_size, int stride, int padding, int output_padding,
                                          int length_out, cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// Ensure tensor is contiguous and on CUDA
torch::Tensor ensure_contiguous_cuda(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");
    return tensor.contiguous();
}

// ===================== conv_transpose1d_forward =====================
torch::Tensor conv_transpose1d_forward(torch::Tensor input, torch::Tensor weight,
                                       c10::optional<torch::Tensor> bias, int64_t stride,
                                       int64_t padding, int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype must be float32");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
        bias_contig = bias.value().contiguous();
    }

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t length_in = input_contig.size(2);
    int64_t out_channels = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == in_channels, "weight first dimension must match input channels");

    int64_t output_length = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    auto output = torch::empty({batch, out_channels, output_length}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(length_in),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        static_cast<int>(output_length),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv_transpose1d kernel launch failed");

    return output;
}

// ===================== conv1d_forward =====================
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight,
                             c10::optional<torch::Tensor> bias, int64_t stride,
                             int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype must be float32");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
        bias_contig = bias.value().contiguous();
    }

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(1) == in_channels, "weight second dimension must match input channels");

    int64_t output_length = (length + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    auto output = torch::empty({batch, out_channels, output_length}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_length),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv1d kernel launch failed");

    return output;
}

// ===================== sum_dim_forward =====================
torch::Tensor sum_dim_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");

    int64_t dim_normalized = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim_normalized >= 0 && dim_normalized < input.dim(), "dim out of range");
    TORCH_CHECK(dim_normalized == 1, "sum_dim_forward currently only supports dim=1");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t length = input.size(2);

    auto output = torch::empty({batch, length}, input.options());

    long long in_stride_batch = input.stride(0);
    long long in_stride_channel = input.stride(1);
    long long in_stride_length = input.stride(2);
    long long out_stride_batch = output.stride(0);
    long long out_stride_length = output.stride(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                     1,
                     static_cast<int>(batch), static_cast<int>(channels), static_cast<int>(length),
                     in_stride_batch, in_stride_channel, in_stride_length,
                     out_stride_batch, out_stride_length,
                     stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sum_dim kernel launch failed");

    return output;
}

// ===================== Register bindings =====================
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "1D transposed convolution using custom CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("conv1d_forward", &conv1d_forward,
          "1D convolution using custom CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("sum_dim_forward", &sum_dim_forward,
          "Sum over dimension (currently only dim=1)",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
