#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#include "../binding_registry.h"

// Declare launchers
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_tan_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

namespace {

torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_abs: input must be float32");
    TORCH_CHECK(x.numel() > 0, "custom_abs: input must have at least one element");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_abs_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs: CUDA error: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_tan(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_tan: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_tan: input must be float32");
    TORCH_CHECK(x.numel() > 0, "custom_tan: input must have at least one element");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_tan_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_tan: CUDA error: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Absolute value of a CUDA float32 tensor", py::arg("x"));
    m.def("custom_tan", &custom_tan, "Tangent of a CUDA float32 tensor", py::arg("x"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
