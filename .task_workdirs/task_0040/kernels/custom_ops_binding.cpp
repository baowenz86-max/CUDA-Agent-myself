#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

extern "C" void roll_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int M, int K, int N,
    cudaStream_t stream);

// Helper to check CUDA tensor and dtype
static void check_cuda_float(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

// conv_transpose1d_forward(input, weight, bias, stride=2, padding=1, output_padding=1)
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    if (bias.defined() && bias.numel() > 0) {
        check_cuda_float(bias, "bias");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    } else {
        bias = torch::Tensor(); // ensure null
    }

    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C_in, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(stride >= 1 && padding >= 0 && output_padding >= 0, "stride/padding must be non-negative");

    int B = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1);
    int K = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight first dim must match input channels");
    if (bias.defined()) {
        TORCH_CHECK(bias.numel() == C_out, "bias size must match output channels");
    }

    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    // Make input and weight contiguous (cuDNN requires contiguous)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    auto output = torch::empty({B, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)output_padding,
        stream
    );

    return output;
}

// bias_add_forward(input, bias)
torch::Tensor bias_add_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(bias.dim() == 2 && bias.size(1) == 1, "bias must be [C,1]");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel size must match input");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int B = input.size(0);
    int C = input.size(1);
    int L = input.size(2);

    auto output = torch::empty_like(input);

    // We can handle non-contiguous input/output by passing strides.
    // For simplicity, if not contiguous, make contiguous copies.
    auto input_contig = input.contiguous();
    auto output_contig = output.contiguous();

    long long in_stride_b = input_contig.stride(0);
    long long in_stride_c = input_contig.stride(1);
    long long in_stride_l = input_contig.stride(2);
    long long out_stride_b = output_contig.stride(0);
    long long out_stride_c = output_contig.stride(1);
    long long out_stride_l = output_contig.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_forward_launcher(
        input_contig.data_ptr<float>(),
        bias.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l,
        stream
    );

    if (!output.is_contiguous()) {
        output.copy_(output_contig);
    }
    return output;
}

// roll_forward(input, shifts=1, dims=2)
torch::Tensor roll_forward(torch::Tensor input, int64_t shifts, int64_t dims) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(dims == 2, "only dims=2 supported");
    TORCH_CHECK(shifts == 1, "only shifts=1 supported");

    int B = input.size(0);
    int C = input.size(1);
    int L = input.size(2);

    auto output = torch::empty_like(input);

    auto input_contig = input.contiguous();
    auto output_contig = output.contiguous();

    long long in_stride_b = input_contig.stride(0);
    long long in_stride_c = input_contig.stride(1);
    long long in_stride_l = input_contig.stride(2);
    long long out_stride_b = output_contig.stride(0);
    long long out_stride_c = output_contig.stride(1);
    long long out_stride_l = output_contig.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    roll_forward_launcher(
        input_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l,
        stream
    );

    if (!output.is_contiguous()) {
        output.copy_(output_contig);
    }
    return output;
}

// linear_forward(input, weight, bias)
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    if (bias.defined() && bias.numel() > 0) {
        check_cuda_float(bias, "bias");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    } else {
        bias = torch::Tensor();
    }

    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    // Flatten input to [M, K]
    auto input_2d = input.reshape({input.size(0), -1}).contiguous();
    int M = input_2d.size(0);
    int K = input_2d.size(1);
    int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "weight second dim must match input feature dim");
    if (bias.defined()) {
        TORCH_CHECK(bias.numel() == N, "bias size must match output features");
    }

    auto output = torch::empty({M, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    linear_forward_launcher(
        input_2d.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, K, N,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 2, py::arg("padding") = 1, py::arg("output_padding") = 1);
    m.def("bias_add_forward", &bias_add_forward,
          "Bias add forward",
          py::arg("input"), py::arg("bias"));
    m.def("roll_forward", &roll_forward,
          "Roll forward",
          py::arg("input"), py::arg("shifts") = 1, py::arg("dims") = 2);
    m.def("linear_forward", &linear_forward,
          "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
