#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Declare CUDA launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void pixel_shuffle_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int upscale_factor,
    cudaStream_t stream);

extern "C" void deconv_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding, int64_t output_padding,
    cudaStream_t stream);

namespace {

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor deconv_forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (C_in, C_out, kH, kW)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 supported");
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Only float32 bias supported");
    }

    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    auto bias_contig = (bias.defined() && bias.numel() > 0) ? (bias.is_contiguous() ? bias : bias.contiguous()) : torch::Tensor();

    int64_t N = x_contig.size(0);
    int64_t C_in = x_contig.size(1);
    int64_t H_in = x_contig.size(2);
    int64_t W_in = x_contig.size(3);
    int64_t C_out = weight_contig.size(1);
    int64_t kH = weight_contig.size(2);
    int64_t kW = weight_contig.size(3);

    TORCH_CHECK(C_in == weight_contig.size(0), "Input channels must match weight in_channels");
    TORCH_CHECK(kH == kernel_size && kW == kernel_size, "Kernel size mismatch");

    int64_t H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int64_t W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions");

    auto output = torch::empty({N, C_out, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* input_ptr = x_contig.data_ptr<float>();
    const float* weight_ptr = weight_contig.data_ptr<float>();
    const float* bias_ptr = bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    deconv_forward_launcher(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        H_out, W_out,
        stride, padding, output_padding,
        stream);

    return output;
}

torch::Tensor pixel_shuffle_forward_impl(const torch::Tensor& x, int64_t upscale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C*r^2, H, W)");

    TORCH_CHECK(upscale_factor == 2 || upscale_factor == 3 || upscale_factor == 4,
                "Only upscale_factor 2, 3, or 4 supported");

    auto x_contig = x.is_contiguous() ? x : x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C_in = x_contig.size(1);
    int64_t H_in = x_contig.size(2);
    int64_t W_in = x_contig.size(3);
    int64_t r = upscale_factor;

    TORCH_CHECK(C_in % (r * r) == 0, "Channel dimension must be divisible by r^2");
    int64_t C_out = C_in / (r * r);
    int64_t H_out = H_in * r;
    int64_t W_out = W_in * r;

    auto output = torch::empty({N, C_out, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pixel_shuffle_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_out, H_in, W_in, H_out, W_out,
        static_cast<int>(upscale_factor),
        stream);

    return output;
}

torch::Tensor relu_forward_impl(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream);

    return output;
}

} // anonymous namespace

torch::Tensor deconv_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    return deconv_forward_impl(x, weight, bias, kernel_size, stride, padding, output_padding);
}

torch::Tensor pixel_shuffle_forward(torch::Tensor x, int64_t upscale_factor) {
    return pixel_shuffle_forward_impl(x, upscale_factor);
}

torch::Tensor relu_forward(torch::Tensor x) {
    return relu_forward_impl(x);
}

void register_custom_ops(pybind11::module& m) {
    m.def("deconv_forward", &deconv_forward, "ConvTranspose2d forward (custom CUDA kernel)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("pixel_shuffle_forward", &pixel_shuffle_forward, "PixelShuffle forward",
          py::arg("x"), py::arg("upscale_factor"));
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
