#include <torch/types.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose2d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    cudaStream_t stream);

extern "C" void clamp_forward_launcher(
    const float* input, float* output, int n, float min_val, float max_val,
    cudaStream_t stream);

extern "C" void leaky_relu_forward_launcher(
    const float* input, float* output, int n, float negative_slope,
    cudaStream_t stream);

extern "C" void triu_forward_launcher(
    const float* input, float* output, int total_elements, int H, int W, int diagonal,
    cudaStream_t stream);

extern "C" void add_forward_launcher(
    const float* input, const float* bias, float* output,
    int N, int C, int HW,
    cudaStream_t stream);

// Helper to validate common tensor requirements
void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ConvTranspose2d forward wrapper
at::Tensor conv_transpose2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t output_padding_h, int64_t output_padding_w) {
    check_cuda_float_contiguous(input, "input");
    check_cuda_float_contiguous(weight, "weight");
    check_cuda_float_contiguous(bias, "bias");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in, C_out, kH, kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (C_out)");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions according to PyTorch formula
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kH + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kW + output_padding_w;

    TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output size");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must match out_channels");

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(padding_h), static_cast<int>(padding_w),
        static_cast<int>(output_padding_h), static_cast<int>(output_padding_w),
        stream);

    return output;
}

at::Tensor clamp_forward(const at::Tensor& input, double min_val, double max_val) {
    check_cuda_float_contiguous(input, "input");
    auto output = torch::empty_like(input);
    int n = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), n,
                           static_cast<float>(min_val), static_cast<float>(max_val), stream);
    return output;
}

at::Tensor leaky_relu_forward(const at::Tensor& input, double negative_slope) {
    check_cuda_float_contiguous(input, "input");
    auto output = torch::empty_like(input);
    int n = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), n,
                                static_cast<float>(negative_slope), stream);
    return output;
}

at::Tensor triu_forward(const at::Tensor& input, int64_t diagonal) {
    check_cuda_float_contiguous(input, "input");
    TORCH_CHECK(input.dim() >= 2, "triu input must have at least 2 dimensions");
    int H = input.size(-2);
    int W = input.size(-1);
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), total_elements,
                          H, W, static_cast<int>(diagonal), stream);
    return output;
}

at::Tensor add_forward(const at::Tensor& input, const at::Tensor& bias) {
    check_cuda_float_contiguous(input, "input");
    check_cuda_float_contiguous(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1,
                "bias must be shape (C, 1, 1)");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    TORCH_CHECK(bias.size(0) == C, "bias channel dimension must match input channels");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_forward_launcher(input.data_ptr<float>(), bias.data_ptr<float>(),
                         output.data_ptr<float>(), N, C, HW, stream);
    return output;
}

// Register operators in torch.ops namespace
TORCH_LIBRARY(cuda_extension, m) {
    m.def("conv_transpose2d_forward(Tensor input, Tensor weight, Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w) -> Tensor",
          &conv_transpose2d_forward);
    m.def("clamp_forward(Tensor input, float min, float max) -> Tensor",
          &clamp_forward);
    m.def("leaky_relu_forward(Tensor input, float negative_slope) -> Tensor",
          &leaky_relu_forward);
    m.def("triu_forward(Tensor input, int diagonal) -> Tensor",
          &triu_forward);
    m.def("add_forward(Tensor input, Tensor bias) -> Tensor",
          &add_forward);
}

// Also register via REGISTER_BINDING for pybind access (optional)
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "ConvTranspose2d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride_h"), pybind11::arg("stride_w"),
          pybind11::arg("padding_h"), pybind11::arg("padding_w"),
          pybind11::arg("output_padding_h"), pybind11::arg("output_padding_w"));
    m.def("clamp_forward", &clamp_forward, "Clamp forward",
          pybind11::arg("input"), pybind11::arg("min"), pybind11::arg("max"));
    m.def("leaky_relu_forward", &leaky_relu_forward, "Leaky ReLU forward",
          pybind11::arg("input"), pybind11::arg("negative_slope"));
    m.def("triu_forward", &triu_forward, "Triu forward",
          pybind11::arg("input"), pybind11::arg("diagonal"));
    m.def("add_forward", &add_forward, "Add bias forward",
          pybind11::arg("input"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
