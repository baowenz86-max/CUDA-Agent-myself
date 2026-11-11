#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
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
extern "C" void conv_transpose_forward_launcher(float* output, const float* input, const float* weight,
                                                const float* bias, int B, int C_in, int L_in,
                                                int C_out, int K, int stride, int padding,
                                                int output_padding, cudaStream_t stream);

namespace {

void checkCuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

torch::Tensor conv_transpose_op_impl(torch::Tensor input, torch::Tensor weight,
                                     c10::optional<torch::Tensor> conv_bias, int64_t stride,
                                     int64_t padding, int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (conv_bias.has_value() && conv_bias.value().defined() && conv_bias.value().numel() > 0) {
        TORCH_CHECK(conv_bias.value().is_cuda(), "conv_bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (conv_bias.has_value() && conv_bias.value().defined() && conv_bias.value().numel() > 0) {
        TORCH_CHECK(conv_bias.value().is_contiguous(), "conv_bias must be contiguous");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (conv_bias.has_value() && conv_bias.value().defined() && conv_bias.value().numel() > 0) {
        TORCH_CHECK(conv_bias.value().dtype() == torch::kFloat32, "conv_bias must be float32");
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
    // bias shape check: should be [C_out]
    if (conv_bias.has_value() && conv_bias.value().defined() && conv_bias.value().numel() > 0) {
        TORCH_CHECK(conv_bias.value().dim() == 1 && conv_bias.value().size(0) == C_out,
                    "conv bias must be shape [out_channels]");
    }

    auto output = torch::empty({B, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* bias_ptr = nullptr;
    if (conv_bias.has_value() && conv_bias.value().defined() && conv_bias.value().numel() > 0) {
        bias_ptr = conv_bias.value().data_ptr<float>();
    }

    conv_transpose_forward_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                                    weight.data_ptr<float>(), bias_ptr,
                                    static_cast<int>(B), static_cast<int>(C_in), static_cast<int>(L_in),
                                    static_cast<int>(C_out), static_cast<int>(kernel_size),
                                    static_cast<int>(stride), static_cast<int>(padding),
                                    static_cast<int>(output_padding), stream);
    checkCuda(cudaGetLastError(), "conv_transpose_forward_launcher failed");
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
                              bias.data_ptr<float>(), static_cast<int>(B),
                              static_cast<int>(C), static_cast<int>(L), stream);
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
                                     static_cast<int>(B), static_cast<int>(C),
                                     static_cast<int>(L), stream);
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
                           static_cast<float>(scaling_factor), static_cast<int>(total), stream);
    checkCuda(cudaGetLastError(), "scale_forward_launcher failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose_op", &conv_transpose_op_impl,
          "ConvTranspose1d forward (custom CUDA kernel)",
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