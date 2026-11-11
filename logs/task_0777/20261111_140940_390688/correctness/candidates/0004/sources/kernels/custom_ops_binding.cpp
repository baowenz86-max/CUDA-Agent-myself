#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_launcher(
    const float* x,
    float* out,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream);

extern "C" void custom_softmax_launcher(
    const float* x,
    float* out,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream);

torch::Tensor custom_mean_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    int ndim = x.dim();
    // Normalize negative dim
    if (dim < 0) dim += ndim;

    // Prepare shape, strides, out_shape vectors
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    std::vector<int64_t> out_shape;
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        strides[i] = x.stride(i);
        if (i != dim) out_shape.push_back(x.size(i));
    }

    // Allocate output tensor (contiguous)
    auto out = torch::empty(out_shape, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_mean_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        shape.data(),
        strides.data(),
        out_shape.data(),
        ndim,
        dim,
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_mean kernel launch failed");

    return out;
}

torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    int ndim = x.dim();
    if (dim < 0) dim += ndim;

    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    std::vector<int64_t> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        strides[i] = x.stride(i);
        out_shape[i] = x.size(i);
    }

    auto out = torch::empty(out_shape, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        shape.data(),
        strides.data(),
        out_shape.data(),
        ndim,
        dim,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_softmax kernel launch failed");

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean", &custom_mean_forward, "Custom mean along dimension",
          py::arg("x"), py::arg("dim"));
    m.def("custom_softmax", &custom_softmax_forward, "Custom softmax along dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
