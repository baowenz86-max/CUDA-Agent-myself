#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    float* y,
    const float* x,
    int64_t numel,
    int ndim,
    const int64_t* shape,
    const int64_t* x_strides,
    const int64_t* y_strides,
    cudaStream_t stream);

extern "C" bool tensor_equal_launcher(
    const float* a,
    const float* b,
    int64_t numel,
    cudaStream_t stream);

// ReLU forward: y = max(0, x), supports arbitrary strides and offsets
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int64_t numel = x.numel();
    if (numel == 0) {
        return torch::empty_like(x);
    }

    int ndim = x.dim();
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> x_strides(ndim);
    std::vector<int64_t> y_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        x_strides[i] = x.stride(i);
    }

    torch::Tensor y;
    if (x.is_non_overlapping_and_dense()) {
        y = torch::empty_strided(x.sizes(), x.strides(), x.options());
    } else {
        y = torch::empty(x.sizes(), x.options());
    }

    for (int i = 0; i < ndim; ++i) {
        y_strides[i] = y.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        ndim,
        shape.data(),
        x_strides.data(),
        y_strides.data(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_forward_launcher failed: ", cudaGetErrorString(err));

    return y;
}

// Tensor equal: check if two tensors have same shape, dtype, device, and values
bool tensor_equal(torch::Tensor a, torch::Tensor b) {
    if (!a.is_cuda() || !b.is_cuda()) return false;
    if (a.scalar_type() != torch::kFloat32 || b.scalar_type() != torch::kFloat32) return false;
    if (!a.sizes().equals(b.sizes())) return false;

    int64_t numel = a.numel();
    if (numel == 0) return true;

    a = a.contiguous();
    b = b.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bool result = tensor_equal_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tensor_equal_launcher failed: ", cudaGetErrorString(err));

    return result;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward (max(0, x))",
          py::arg("x"));
    m.def("tensor_equal", &tensor_equal, "Compare two tensors for exact equality",
          py::arg("a"), py::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
