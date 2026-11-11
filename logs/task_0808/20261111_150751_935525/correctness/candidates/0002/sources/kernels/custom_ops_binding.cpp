#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void constant_pad_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    float value,
    cudaStream_t stream);

extern "C" void stack_duplicate_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    cudaStream_t stream);

torch::Tensor constant_pad_forward(torch::Tensor x, int64_t pad, double value) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // In this contract, pad is a scalar representing equal padding on all four sides.
    int64_t pad_left = pad;
    int64_t pad_right = pad;
    int64_t pad_top = pad;
    int64_t pad_bottom = pad;

    int64_t out_H = H + pad_top + pad_bottom;
    int64_t out_W = W + pad_left + pad_right;
    auto output = torch::empty({N, C, out_H, out_W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    constant_pad_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        static_cast<float>(value),
        stream);

    return output;
}

torch::Tensor stack_duplicate_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    auto output = torch::empty({N, 2, C, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    stack_duplicate_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad", &constant_pad_forward,
          "constant_pad(x: Tensor, pad: int, value: float) -> Tensor",
          pybind11::arg("x"), pybind11::arg("pad"), pybind11::arg("value"));
    m.def("stack_duplicate", &stack_duplicate_forward,
          "stack_duplicate(x: Tensor) -> Tensor",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
