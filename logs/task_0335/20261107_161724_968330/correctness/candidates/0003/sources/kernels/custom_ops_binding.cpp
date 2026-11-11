#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
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

extern "C" void conv_transpose1d_launcher(float* output,
                                          const float* input,
                                          const float* weight,
                                          const float* bias,
                                          int batch,
                                          int in_channels,
                                          int out_channels,
                                          int length_in,
                                          int length_out,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          cudaStream_t stream);

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

// ------------------- conv_transpose1d_forward -------------------
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
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    conv_transpose1d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        batch,
        in_channels,
        out_channels,
        length_in,
        length_out,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("conv_transpose1d kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
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
