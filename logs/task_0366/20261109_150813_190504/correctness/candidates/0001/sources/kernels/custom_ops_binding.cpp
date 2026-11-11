#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" bool custom_equal_cpu_int64_launcher(
    const int64_t* x,
    const int64_t* y,
    int64_t numel,
    cudaStream_t stream);

extern "C" bool custom_sin_cuda_int64_launcher(
    const int64_t* cpu_input,
    int64_t numel,
    float* cuda_output,
    cudaStream_t stream);

bool custom_equal_cpu_int64(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(!x.is_cuda(), "custom_equal_cpu_int64: x must be a CPU tensor");
    TORCH_CHECK(!y.is_cuda(), "custom_equal_cpu_int64: y must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "custom_equal_cpu_int64: x must be int64");
    TORCH_CHECK(y.dtype() == torch::kInt64, "custom_equal_cpu_int64: y must be int64");
    TORCH_CHECK(x.is_contiguous(), "custom_equal_cpu_int64: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "custom_equal_cpu_int64: y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "custom_equal_cpu_int64: x and y must have the same shape");

    int64_t numel = x.numel();
    if (numel == 0) {
        return true;
    }

    // Use the current CUDA stream (unused by launcher, but required for signature consistency)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    return custom_equal_cpu_int64_launcher(
        x.data_ptr<int64_t>(),
        y.data_ptr<int64_t>(),
        numel,
        stream);
}

torch::Tensor custom_sin_cuda_int64(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "custom_sin_cuda_int64: x must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "custom_sin_cuda_int64: x must be int64");
    TORCH_CHECK(x.is_contiguous(), "custom_sin_cuda_int64: x must be contiguous");

    int64_t numel = x.numel();
    auto output = torch::empty_like(x, x.options().dtype(torch::kFloat32).device(torch::kCUDA));
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bool success = custom_sin_cuda_int64_launcher(
        x.data_ptr<int64_t>(),
        numel,
        output.data_ptr<float>(),
        stream);

    TORCH_CHECK(success, "custom_sin_cuda_int64: launcher failed");

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_equal_cpu_int64", &custom_equal_cpu_int64,
          "Compare two int64 CPU tensors element-wise (torch.equal)",
          py::arg("x"), py::arg("y"));
    m.def("custom_sin_cuda_int64", &custom_sin_cuda_int64,
          "Compute sin of int64 CPU tensor, producing float32 CUDA tensor",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
