#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../binding_registry.h"

extern "C" void custom_sum_dim1_launcher(
    const float* x,
    float* out,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void custom_fmod_launcher(
    const float* x,
    float* out,
    int64_t n,
    double divisor,
    cudaStream_t stream);

torch::Tensor custom_sum_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_sum_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_sum_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_sum_dim1: input must be 4D [B, C, H, W]");
    TORCH_CHECK(x.is_contiguous(), "custom_sum_dim1: input must be contiguous");

    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    auto out = torch::empty({B, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sum_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sum_dim1 launch failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor custom_fmod(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "custom_fmod: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_fmod: input must be float32");
    TORCH_CHECK(x.dim() > 0, "custom_fmod: input cannot be empty");
    TORCH_CHECK(x.is_contiguous(), "custom_fmod: input must be contiguous");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_fmod_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel(),
        divisor,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_fmod launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_sum_dim1", &custom_sum_dim1, "Sum over dimension 1", py::arg("x"));
    m.def("custom_fmod", &custom_fmod, "Elementwise fmod with scalar divisor", py::arg("x"), py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
