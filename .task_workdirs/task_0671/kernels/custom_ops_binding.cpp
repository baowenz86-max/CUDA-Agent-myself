#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(const float* input,
                                     float* output,
                                     int64_t n,
                                     cudaStream_t stream);

extern "C" void custom_floor_launcher(const float* input,
                                       float* output,
                                       int64_t n,
                                       cudaStream_t stream);

// Wrapper for custom_abs
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_abs: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t n = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(x.data_ptr<float>(), output.data_ptr<float>(), n, stream);

    return output;
}

// Wrapper for custom_floor
torch::Tensor custom_floor(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_floor: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_floor: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_floor: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t n = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_floor_launcher(x.data_ptr<float>(), output.data_ptr<float>(), n, stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Custom absolute value kernel", py::arg("x"));
    m.def("custom_floor", &custom_floor, "Custom floor kernel", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
