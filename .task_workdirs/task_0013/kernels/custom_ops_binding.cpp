#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Extern declarations for launchers defined in custom_ops.cu
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void rsqrt_eps_launcher(
    const float* input,
    float* output,
    int numel,
    float eps,
    cudaStream_t stream);

extern "C" void softmax_lastdim_launcher(
    const float* input,
    float* output,
    int outer_size,
    int inner_size,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// conv_transpose1d_forward
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight first dimension must match input channels");

    // PyTorch output length formula for ConvTranspose1d (dilation=1, groups=1)
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "calculated output length must be positive");

    auto output = torch::empty({N, C_out, L_out}, input.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt->defined()) {
        torch::Tensor bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "bias must be 1D with size C_out");
        bias_ptr = bias.data_ptr<float>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// rsqrt_eps_forward
// -----------------------------------------------------------------------------
torch::Tensor rsqrt_eps_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rsqrt_eps_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        1e-12f,
        stream);
    return output;
}

// -----------------------------------------------------------------------------
// softmax_dim_forward (dim = -1 only)
// -----------------------------------------------------------------------------
torch::Tensor softmax_dim_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == -1 || dim == input.dim() - 1, "only dim=-1 is supported");

    auto output = torch::empty_like(input);
    int64_t inner_size = input.size(-1);
    int64_t outer_size = input.numel() / inner_size;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_lastdim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(outer_size),
        static_cast<int>(inner_size),
        stream);
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward (custom CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("output_padding") = 0);

    m.def("rsqrt_eps_forward", &rsqrt_eps_forward,
          "rsqrt(abs(x) + 1e-12) forward (custom CUDA)",
          pybind11::arg("input"));

    m.def("softmax_dim_forward", &softmax_dim_forward,
          "Softmax along last dimension (custom CUDA)",
          pybind11::arg("input"),
          pybind11::arg("dim") = -1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
