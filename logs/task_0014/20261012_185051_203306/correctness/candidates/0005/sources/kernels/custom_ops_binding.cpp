#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_bias_free_launcher(
    const float* input,
    const float* weight,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void mask_not_equal_launcher(
    const float* input,
    uint8_t* mask,
    int numel,
    float mask_val,
    cudaStream_t stream);

extern "C" void masked_select_launcher(
    const float* input,
    const uint8_t* mask,
    float* output,
    long long* count,
    int numel,
    cudaStream_t stream);

// Helper to get current stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ConvTranspose1d bias-free forward
static torch::Tensor conv_transpose1d_bias_free_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t kernel_size) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");

    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1);
    // Validate weight first dimension matches C_in
    TORCH_CHECK(weight.size(0) == C_in, "weight C_in dimension mismatch");
    TORCH_CHECK(weight.size(2) == kernel_size, "weight kernel size mismatch");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "computed L_out must be positive");

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = getCurrentStream();
    conv_transpose1d_bias_free_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, static_cast<int>(L_out),
        static_cast<int>(kernel_size), static_cast<int>(stride),
        static_cast<int>(padding), static_cast<int>(output_padding),
        stream);

    return output;
}

// Clamp forward
static torch::Tensor clamp_forward(
    torch::Tensor input,
    double min_val,
    double max_val) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int numel = static_cast<int>(input.numel());

    cudaStream_t stream = getCurrentStream();
    clamp_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    return output;
}

// Mask not equal forward (returns bool tensor)
static torch::Tensor mask_not_equal_forward(
    torch::Tensor input,
    double mask_val) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    int numel = static_cast<int>(input.numel());

    cudaStream_t stream = getCurrentStream();
    mask_not_equal_launcher(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        numel,
        static_cast<float>(mask_val),
        stream);

    return output;
}

// Masked select from existing mask
static torch::Tensor masked_select_from_mask_forward(
    torch::Tensor input,
    torch::Tensor mask) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
    TORCH_CHECK(input.numel() == mask.numel(), "input and mask must have same numel");

    int numel = static_cast<int>(input.numel());
    auto output = torch::empty({numel}, input.options());
    auto count = torch::zeros({1}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = getCurrentStream();
    masked_select_launcher(
        input.data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        reinterpret_cast<long long*>(count.data_ptr<int64_t>()),
        numel,
        stream);

    // Synchronize to read count on CPU
    cudaStreamSynchronize(stream);
    int64_t valid_count = count.cpu().item<int64_t>();
    TORCH_CHECK(valid_count <= numel, "internal error: count exceeds numel");
    return output.narrow(0, 0, valid_count).contiguous();
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_bias_free_forward", &conv_transpose1d_bias_free_forward,
          "ConvTranspose1d bias-free forward",
          pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("stride"), pybind11::arg("padding"),
          pybind11::arg("output_padding"), pybind11::arg("kernel_size"));
    m.def("clamp_forward", &clamp_forward,
          "Clamp forward",
          pybind11::arg("input"), pybind11::arg("min_val"), pybind11::arg("max_val"));
    m.def("mask_not_equal_forward", &mask_not_equal_forward,
          "Mask not equal forward",
          pybind11::arg("input"), pybind11::arg("mask_val"));
    m.def("masked_select_from_mask_forward", &masked_select_from_mask_forward,
          "Masked select from mask forward",
          pybind11::arg("input"), pybind11::arg("mask"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
