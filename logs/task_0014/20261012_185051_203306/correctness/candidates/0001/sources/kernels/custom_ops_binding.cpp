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
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

extern "C" void clamp_launcher(
    const float* in,
    float* out,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void mask_not_equal_launcher(
    const float* in,
    bool* mask,
    int total,
    float mask_val,
    cudaStream_t stream);

extern "C" void masked_select_launcher(
    const float* in,
    const bool* mask,
    float* out,
    int total,
    int* count,
    cudaStream_t stream);

// Wrapper for conv_transpose1d without bias
torch::Tensor conv_transpose1d_bias_free_forward(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int C_out = weight.size(1);
    int kernel_size = weight.size(2);
    // Hardcoded stride, padding, output_padding as in model_new.py usage
    int stride = 2;
    int padding = 1;
    int output_padding = 1;
    int L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    TORCH_CHECK(weight.size(0) == C_in, "weight first dimension must match input channels");

    auto out = torch::empty({N, C_out, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_bias_free_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, output_padding,
        stream);
    return out;
}

// Wrapper for clamp (torch.clamp)
torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);
    return out;
}

// Wrapper for mask_not_equal (debug only)
torch::Tensor mask_not_equal_forward(torch::Tensor x, double mask_val) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto mask = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mask_not_equal_launcher(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        x.numel(),
        static_cast<float>(mask_val),
        stream);
    return mask;
}

// Helper to perform masked_select given a precomputed mask (debug only)
torch::Tensor masked_select_from_mask_forward(torch::Tensor x, torch::Tensor mask) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(x.numel() == mask.numel(), "x and mask must have same number of elements");

    int total = x.numel();
    // Allocate maximum possible output size (total elements)
    auto out = torch::empty({total}, x.options());
    // Allocate device count tensor with one element initialized to 0
    auto count = torch::zeros({1}, x.options().dtype(torch::kInt));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_select_launcher(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        total,
        count.data_ptr<int>(),
        stream);

    // Copy count to CPU to get actual number of selected elements
    int actual_count = count.cpu().item<int>();
    return out.narrow(0, 0, actual_count);
}

// Wrapper for masked_select that combines mask creation and selection
torch::Tensor masked_select_forward(torch::Tensor x, double mask_val) {
    // Create mask on GPU
    torch::Tensor mask = mask_not_equal_forward(x, mask_val);
    return masked_select_from_mask_forward(x, mask);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_bias_free_forward", &conv_transpose1d_bias_free_forward,
          "ConvTranspose1d without bias",
          py::arg("x"), py::arg("weight"));
    m.def("clamp_forward", &clamp_forward,
          "Clamp forward",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
    m.def("mask_not_equal_forward", &mask_not_equal_forward,
          "Mask not equal (debug)",
          py::arg("x"), py::arg("mask_val"));
    m.def("masked_select_from_mask_forward", &masked_select_from_mask_forward,
          "Masked select from mask (debug)",
          py::arg("x"), py::arg("mask"));
    m.def("masked_select_forward", &masked_select_forward,
          "Masked select combined",
          py::arg("x"), py::arg("mask_val"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
