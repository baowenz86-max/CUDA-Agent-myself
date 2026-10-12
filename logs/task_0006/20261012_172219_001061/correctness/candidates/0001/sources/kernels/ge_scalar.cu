#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

__global__ void ge_scalar_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    float scalar,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = idx; i < total_elements; i += stride) {
        output[i] = (input[i] >= scalar);
    }
}

torch::Tensor ge_scalar_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));

    long total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ge_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<bool>(),
        (float)scalar,
        total_elements
    );

    return output;
}

void register_ge_scalar(py::module& m) {
    m.def("ge_scalar_forward", &ge_scalar_forward,
          "Greater than or equal scalar forward",
          py::arg("input"), py::arg("scalar"));
}

REGISTER_BINDING(ge_scalar, register_ge_scalar);
