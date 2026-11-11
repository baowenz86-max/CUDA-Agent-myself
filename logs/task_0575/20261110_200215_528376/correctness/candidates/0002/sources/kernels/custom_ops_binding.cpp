#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int C,
    int H,
    int W,
    int64_t stride_b,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    cudaStream_t stream
);

extern "C" void round_tensor_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream
);

// Wrapper for max over dim=1
std::tuple<torch::Tensor, torch::Tensor> max_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (batch, C, H, W)");

    int batch = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto values = torch::empty({batch, H, W}, x.options());
    auto indices = torch::empty({batch, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_dim1_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

// Wrapper for round (half-to-even)
torch::Tensor round_tensor(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Launcher assumes contiguous 1D memory, so make contiguous if necessary.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    round_tensor_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "round_tensor kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_dim1", &max_dim1, "Max over dim=1 returning (values, indices)",
          py::arg("x"));
    m.def("round_tensor", &round_tensor, "Round tensor half-to-even",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
