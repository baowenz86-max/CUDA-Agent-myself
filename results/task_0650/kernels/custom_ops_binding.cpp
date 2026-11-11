#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void isnan_forward_launcher(const float* input, uint8_t* output, int64_t numel, cudaStream_t stream);
extern "C" void gather_forward_launcher(
    const float* input,
    const int64_t* index,
    float* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    cudaStream_t stream
);

// PyTorch binding for isnan_forward
torch::Tensor isnan_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure contiguous (launcher assumes contiguous)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto output = torch::empty_like(x, x.options().dtype(torch::kBool));
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // bool tensor storage is 1 byte per element; use uint8_t* for launcher
    isnan_forward_launcher(x.data_ptr<float>(),
                           reinterpret_cast<uint8_t*>(output.data_ptr<bool>()),
                           numel,
                           stream);

    return output;
}

// PyTorch binding for gather_forward
torch::Tensor gather_forward(torch::Tensor x, int64_t dim, torch::Tensor index) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "Index must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(index.dtype() == torch::kInt64, "Index must be int64");

    // Normalize dim to non-negative
    int64_t ndim = x.dim();
    TORCH_CHECK(ndim == index.dim(), "x and index must have same number of dimensions");
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Ensure contiguous (launcher assumes contiguous)
    if (!x.is_contiguous()) x = x.contiguous();
    if (!index.is_contiguous()) index = index.contiguous();

    // Compute sizes for gather launcher
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }
    int64_t dim_size = x.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= index.size(i);
    }
    // Note: torch.gather requires index.size(i) <= x.size(i) for i != dim.

    // Output shape = index shape, dtype float32, device CUDA
    auto output = torch::empty_like(index, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gather_forward_launcher(
        x.data_ptr<float>(),
        index.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        outer_size,
        dim_size,
        inner_size,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("isnan_forward", &isnan_forward, "Check for NaN values element-wise",
          py::arg("x"));
    m.def("gather_forward", &gather_forward, "Gather values along a dimension",
          py::arg("x"), py::arg("dim"), py::arg("index"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
