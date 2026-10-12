// Binding file for custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_bias_free_launcher(
    const float* x,
    const float* weight,
    float* out,
    int N,
    int C_in,
    int C_out,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int64_t N,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void mask_not_equal_launcher(
    const float* input,
    bool* output,
    int64_t N,
    float mask_val,
    cudaStream_t stream);

extern "C" void masked_select_launcher(
    const float* input,
    const bool* mask,
    float* output,
    int64_t N,
    int64_t* count,
    cudaStream_t stream);

// Wrapper functions

// ConvTranspose1d bias-free forward
torch::Tensor conv_transpose1d_bias_free_forward(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");
    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t L_in = x.size(2);
    const int64_t kernel_size = weight.size(2);
    const int64_t C_out = weight.size(1);
    TORCH_CHECK(C_in == weight.size(0), "weight shape mismatch: C_in must match");
    const int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "invalid L_out computed");
    auto out = torch::empty({N, C_out, L_out}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_bias_free_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
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
    return out;
}

// Clamp forward
torch::Tensor clamp_forward(torch::Tensor input, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);
    return output;
}

// Mask not equal forward (returns bool tensor)
torch::Tensor mask_not_equal_forward(torch::Tensor input, double mask_val) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kBool));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mask_not_equal_launcher(
        input.data_ptr<float>(),
        output.data_ptr<bool>(),
        input.numel(),
        static_cast<float>(mask_val),
        stream);
    return output;
}

// Masked select from mask forward
torch::Tensor masked_select_from_mask_forward(torch::Tensor input, torch::Tensor mask) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
    TORCH_CHECK(input.numel() == mask.numel(), "input and mask must have same number of elements");
    const int64_t N = input.numel();
    auto output = torch::empty({N}, input.options());
    auto count = torch::zeros({1}, torch::dtype(torch::kInt64).device(input.device()));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_select_launcher(
        input.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        N,
        count.data_ptr<int64_t>(),
        stream);
    // Synchronize to get count on CPU
    int64_t num_selected = count.item<int64_t>();
    return output.narrow(0, 0, num_selected).contiguous();
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_bias_free_forward", &conv_transpose1d_bias_free_forward, "ConvTranspose1d bias-free forward",
          py::arg("x"), py::arg("weight"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("clamp_forward", &clamp_forward, "Clamp forward",
          py::arg("input"), py::arg("min_val"), py::arg("max_val"));
    m.def("mask_not_equal_forward", &mask_not_equal_forward, "Mask not equal forward",
          py::arg("input"), py::arg("mask_val"));
    m.def("masked_select_from_mask_forward", &masked_select_from_mask_forward, "Masked select from mask forward",
          py::arg("input"), py::arg("mask"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
