#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void hardsigmoid_forward_launcher(const float* input,
                                              float* output,
                                              int64_t numel,
                                              cudaStream_t stream);

torch::Tensor hardsigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardsigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

void register_hardsigmoid(pybind11::module& m) {
    m.def("hardsigmoid_forward", &hardsigmoid_forward, "Hardsigmoid forward",
          py::arg("x"));
}

REGISTER_BINDING(hardsigmoid, register_hardsigmoid);
