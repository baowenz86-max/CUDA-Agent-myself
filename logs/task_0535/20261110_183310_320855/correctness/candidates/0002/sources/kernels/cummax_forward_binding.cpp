// Binding for cummax_forward operator
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void cummax_forward_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t N,
    int64_t M,
    int64_t dim,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> cummax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported");

    int64_t N = x.size(0);
    int64_t M = x.size(1);

    auto values = torch::empty_like(x);
    auto indices = torch::empty(x.sizes(), x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cummax_forward_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N,
        M,
        dim,
        stream
    );

    return std::make_tuple(values, indices);
}

void register_cummax_forward(pybind11::module& m) {
    m.def("cummax_forward", &cummax_forward,
          "Cumulative maximum along dim=0",
          py::arg("x"),
          py::arg("dim"));
}

REGISTER_BINDING(cummax_forward, register_cummax_forward);
