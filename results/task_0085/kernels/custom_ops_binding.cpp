#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void clamp_scalar_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void sub_scalar_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float scalar,
    cudaStream_t stream);

// Wrapper for clamp_scalar
torch::Tensor clamp_scalar(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_scalar_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_scalar kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Wrapper for sub_scalar
torch::Tensor sub_scalar(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sub_scalar_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        static_cast<float>(scalar),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sub_scalar kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_scalar", &clamp_scalar,
          "Clamp tensor elements to [min_val, max_val]",
          pybind11::arg("x"),
          pybind11::arg("min_val"),
          pybind11::arg("max_val"));
    m.def("sub_scalar", &sub_scalar,
          "Subtract scalar from tensor elements",
          pybind11::arg("x"),
          pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
