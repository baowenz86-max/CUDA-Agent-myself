#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void custom_softmax_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

torch::Tensor custom_softmax(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    x = x.contiguous();

    auto output = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        N, C, H, W,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_softmax(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax,
          "Custom softmax along dimension 1 of a 4D tensor",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_softmax, register_custom_softmax);
