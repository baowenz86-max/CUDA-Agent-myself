#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in,
    int C_out, int K, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void rsqrt_eps_launcher(
    const float* input,
    float* output,
    float eps,
    int total,
    cudaStream_t stream);

extern "C" void softmax_lastdim_launcher(
    const float* input,
    float* output,
    int outer,
    int inner,
    cudaStream_t stream);

namespace {

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    std::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int B = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1);
    int K = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight shape mismatch");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C_out, "bias size mismatch");
        bias_ptr = bias.data_ptr<float>();
    }

    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    auto output = torch::empty({B, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        B, C_in, L_in, C_out, K, stride, padding, output_padding,
        stream);

    return output;
}

torch::Tensor rsqrt_eps_forward(torch::Tensor input, double eps)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rsqrt_eps_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(eps),
        input.numel(),
        stream);
    return output;
}

torch::Tensor softmax_lastdim_forward(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int64_t inner = input.size(-1);
    int64_t outer = input.numel() / inner;

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_lastdim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(outer),
        static_cast<int>(inner),
        stream);
    return output;
}

} // namespace

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride") = 1, pybind11::arg("padding") = 0, pybind11::arg("output_padding") = 0);
    m.def("rsqrt_eps_forward", &rsqrt_eps_forward,
          "rsqrt(abs(x)+eps)",
          pybind11::arg("input"), pybind11::arg("eps") = 1e-12);
    m.def("softmax_lastdim_forward", &softmax_lastdim_forward,
          "softmax over last dimension",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
