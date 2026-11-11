// 移除 cudnn 依赖，使用自定义 CUDA kernel 实现 ConvTranspose1d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "../binding_registry.h"

// 声明 .cu 中的 launchers
extern "C" cudaError_t custom_silu_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

extern "C" cudaError_t custom_conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t L_in,
    int64_t L_out,
    int64_t K,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    bool has_bias,
    cudaStream_t stream
);

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                ": " + cudaGetErrorString(err)); \
        } \
    } while (0)

torch::Tensor custom_silu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_silu: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_silu: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_silu: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CUDA_CHECK(custom_silu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    ));
    return output;
}

torch::Tensor custom_conv_transpose1d(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N,C,L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in,C_out,K)");

    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);
    TORCH_CHECK(weight.size(0) == C_in, "weight shape mismatch");
    TORCH_CHECK(stride > 0, "stride > 0");
    TORCH_CHECK(padding >= 0, "padding >= 0");
    TORCH_CHECK(output_padding >= 0 && output_padding < stride, "invalid output_padding");

    bool has_bias = bias.has_value();
    torch::Tensor bias_tensor;
    if (has_bias) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == C_out, "bias size must be C_out");
    }

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (has_bias) bias_contig = bias_tensor.contiguous();

    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "L_out must be positive");

    auto output = torch::empty({N, C_out, L_out}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;

    CUDA_CHECK(custom_conv_transpose1d_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out,
        K, stride, padding, output_padding,
        has_bias,
        stream
    ));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
          "Custom ConvTranspose1d",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("custom_silu", &custom_silu, "Custom SiLU", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
