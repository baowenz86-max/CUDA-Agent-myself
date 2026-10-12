// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_log10_launcher(const float* input, float* output, long long numel, cudaStream_t stream);
extern "C" void custom_lt_launcher(const float* input, bool* output, float threshold, long long numel, cudaStream_t stream);
extern "C" void custom_conv_transpose1d_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding, int dilation, int groups,
    cudaStream_t stream);
extern "C" void custom_add_bias_launcher(
    const float* input, const float* bias, float* output,
    int N, int C, int L, cudaStream_t stream);

static torch::Tensor custom_log10(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(input);
    long long numel = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_log10_launcher(input.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    return output;
}

static torch::Tensor custom_lt(torch::Tensor input, double threshold) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    long long numel = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_lt_launcher(input.data_ptr<float>(), output.data_ptr<bool>(), (float)threshold, numel, stream);
    return output;
}

static torch::Tensor custom_conv_transpose1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
    int64_t groups) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (N, C_in, L)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (C_in, C_out/groups, K)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1) * groups;
    int kernel_size = weight.size(2);
    TORCH_CHECK(C_in == weight.size(0), "Weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "Bias size mismatch");
    TORCH_CHECK(groups >= 1 && C_in % groups == 0 && C_out % groups == 0, "Invalid groups");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && dilation > 0, "Invalid parameters");

    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_conv_transpose1d_launcher(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out,
        kernel_size, (int)stride, (int)padding, (int)output_padding, (int)dilation, (int)groups,
        stream);
    return output;
}

static torch::Tensor custom_add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D");
    TORCH_CHECK(bias.dim() == 2, "Bias must be 2D (C, 1)");
    int N = input.size(0);
    int C = input.size(1);
    int L = input.size(2);
    TORCH_CHECK(bias.size(0) == C, "Bias channels mismatch");
    TORCH_CHECK(bias.size(1) == 1, "Bias must have shape (C, 1)");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_add_bias_launcher(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, C, L, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_log10", &custom_log10, "log10");
    m.def("custom_lt", &custom_lt, "less than");
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d, "conv transpose 1d");
    m.def("custom_add_bias", &custom_add_bias, "add bias");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
