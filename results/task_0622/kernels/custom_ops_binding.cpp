#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream);

extern "C" void unsqueeze_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           int64_t dim,
                                           cudaStream_t stream);

// PyTorch wrapper for abs
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(x.data_ptr<float>(),
                         output.data_ptr<float>(),
                         x.numel(),
                         stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

// PyTorch wrapper for unsqueeze
torch::Tensor unsqueeze_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int64_t input_dim = x.dim();
    TORCH_CHECK(dim >= 0 && dim <= input_dim, "dim out of range");

    // Compute output shape: insert 1 at dim
    std::vector<int64_t> output_shape;
    output_shape.reserve(input_dim + 1);
    for (int64_t i = 0; i < input_dim; ++i) {
        if (i == dim) output_shape.push_back(1);
        output_shape.push_back(x.size(i));
    }
    if (dim == input_dim) output_shape.push_back(1);

    auto output = torch::empty(output_shape, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unsqueeze_forward_launcher(x.data_ptr<float>(),
                               output.data_ptr<float>(),
                               x.numel(),
                               dim,
                               stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("unsqueeze_forward", &unsqueeze_forward, "Unsqueeze operation",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
