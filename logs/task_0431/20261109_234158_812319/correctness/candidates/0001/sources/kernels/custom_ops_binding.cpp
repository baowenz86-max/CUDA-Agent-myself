#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void softplus_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void silu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t H_out, int64_t W_out,
    cudaStream_t stream);

// Softplus wrapper
torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream);
    return output;
}

// SiLU wrapper
torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream);
    return output;
}

// ReflectionPad2d wrapper
torch::Tensor reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom,
        H_out, W_out,
        stream);
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("softplus_forward", &softplus_forward, "Softplus forward", py::arg("x"));
    m.def("silu_forward", &silu_forward, "SiLU forward", py::arg("x"));
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
