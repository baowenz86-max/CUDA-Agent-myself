#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void nan_to_num_launcher(
    const float* x,
    float* y,
    int64_t numel,
    cudaStream_t stream);

torch::Tensor custom_sum_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto sizes = x.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    auto y = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return y;
}

torch::Tensor custom_nan_to_num(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto y = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    nan_to_num_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream);

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_sum_dim1_keepdim", &custom_sum_dim1_keepdim,
          "Sum over dimension 1 (channels) with keepdim=True",
          py::arg("x"));
    m.def("custom_nan_to_num", &custom_nan_to_num,
          "Replace NaN values with zero",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
