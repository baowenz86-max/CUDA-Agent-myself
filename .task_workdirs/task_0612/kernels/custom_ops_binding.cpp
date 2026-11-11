#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers
extern "C" void custom_sum_dim_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t total_output,
    cudaStream_t stream);

extern "C" void custom_expm1_launcher(
    const float* input,
    float* output,
    int64_t total,
    cudaStream_t stream);

torch::Tensor custom_sum_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4-dimensional");
    TORCH_CHECK(dim == 1, "This custom op only supports dim=1 for this task");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({B, H, W}, x.options());
    int64_t total_output = output.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sum_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W, total_output,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sum_dim kernel launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sum_dim kernel execution failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_expm1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto output = torch::empty_like(x);
    int64_t total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_expm1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_expm1 kernel launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_expm1 kernel execution failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_sum_dim", &custom_sum_dim,
          "Sum over dimension 1 of a 4D float32 CUDA tensor",
          py::arg("x"), py::arg("dim"));
    m.def("custom_expm1", &custom_expm1,
          "Elementwise expm1 for float32 CUDA tensor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
