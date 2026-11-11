#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void max_reduce_div_launcher(
    const float* input,
    float* values_out,
    int64_t* indices_out,
    int N,
    int C,
    int H,
    int W,
    double divisor,
    cudaStream_t stream);

// PyTorch binding for fused max reduction along dim=1 and division
py::tuple max_reduce_div_forward(
    torch::Tensor x,
    int64_t dim,
    bool keepdim,
    double divisor)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional [N,C,H,W]");
    TORCH_CHECK(dim == 1, "Only reduction over dim=1 is supported");
    TORCH_CHECK(keepdim, "keepdim must be true for this op");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto N = static_cast<int>(x.size(0));
    auto C = static_cast<int>(x.size(1));
    auto H = static_cast<int>(x.size(2));
    auto W = static_cast<int>(x.size(3));

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_reduce_div_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N,
        C,
        H,
        W,
        divisor,
        stream);

    return py::make_tuple(values, indices);
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_reduce_div_forward", &max_reduce_div_forward,
          "Fused max reduction over dim=1 and division by scalar",
          py::arg("x"),
          py::arg("dim"),
          py::arg("keepdim"),
          py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
