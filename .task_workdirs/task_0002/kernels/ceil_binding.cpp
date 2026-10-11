#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from ceil_kernel.cu
extern "C" void ceil_launcher(
    const float* input,
    float* output,
    int N,
    cudaStream_t stream);

// PyTorch wrapper for ceil
torch::Tensor ceil_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ceil_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream);

    return output;
}

// Registration function
void register_ceil(pybind11::module& m) {
    m.def("ceil_forward", &ceil_forward, "Ceil forward",
          py::arg("input"));
}

REGISTER_BINDING(ceil, register_ceil);
