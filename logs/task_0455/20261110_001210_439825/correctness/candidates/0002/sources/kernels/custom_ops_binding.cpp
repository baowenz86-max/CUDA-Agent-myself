#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t batch,
    int64_t dim_size,
    int64_t height,
    int64_t width,
    cudaStream_t stream);

extern "C" void custom_argmin_dim_launcher(
    const float* input,
    int64_t* indices,
    int64_t batch,
    int64_t dim_size,
    int64_t height,
    int64_t width,
    cudaStream_t stream);

// Helper to resolve dim and ensure contiguous, CUDA float32 tensor
static std::tuple<torch::Tensor, int64_t, int64_t, int64_t, int64_t>
prepare_input(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    
    int64_t ndim = x.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");
    
    // Make contiguous if not already
    x = x.contiguous();
    
    // Assume shape [batch, channels, height, width]
    int64_t batch = x.size(0);
    int64_t dim_size = x.size(dim);
    int64_t height = x.size(2);
    int64_t width = x.size(3);
    
    // For dim=1, output is [batch, height, width]
    // Other dims not supported by current launcher (launcher assumes dim=1)
    TORCH_CHECK(dim == 1, "Only dim=1 supported");
    
    return {x, batch, dim_size, height, width};
}

std::tuple<torch::Tensor, torch::Tensor> custom_max_dim(torch::Tensor x, int64_t dim) {
    auto [x_contig, batch, dim_size, height, width] = prepare_input(x, dim);
    
    auto values = torch::empty({batch, height, width}, x_contig.options());
    auto indices = torch::empty({batch, height, width}, x_contig.options().dtype(torch::kInt64));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_max_dim_launcher(
        x_contig.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch,
        dim_size,
        height,
        width,
        stream);
    
    return std::make_tuple(values, indices);
}

torch::Tensor custom_argmin_dim(torch::Tensor x, int64_t dim) {
    auto [x_contig, batch, dim_size, height, width] = prepare_input(x, dim);
    
    auto indices = torch::empty({batch, height, width}, x_contig.options().dtype(torch::kInt64));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_argmin_dim_launcher(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch,
        dim_size,
        height,
        width,
        stream);
    
    return indices;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim", &custom_max_dim,
          "Custom max over dim",
          py::arg("x"), py::arg("dim"));
    m.def("custom_argmin_dim", &custom_argmin_dim,
          "Custom argmin over dim",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
