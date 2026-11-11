#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declaration of the launcher implemented in custom_ops.cu
extern "C" void max_forward_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t* input_shape,
    int64_t ndim,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream);

// Wrapper function matching the contract: torch.return_types.max max_forward(torch::Tensor x, int64_t dim, bool keepdim)
torch::return_types::max max_forward(torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input dtype must be float32");

    // Ensure contiguous; the launcher assumes contiguous input.
    auto x_contig = x.contiguous();

    int64_t ndim = x_contig.dim();
    TORCH_CHECK(ndim > 0, "Input tensor must have at least one dimension");

    // Normalize dim
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(x_contig.size(i));
        }
    }

    auto values = torch::empty(out_shape, x_contig.options());
    auto indices = torch::empty(out_shape, x_contig.options().dtype(torch::kInt64));

    // Get input shape as int64_t array on host
    std::vector<int64_t> input_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        input_shape[i] = x_contig.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_forward_launcher(
        x_contig.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        input_shape.data(),
        ndim,
        dim,
        keepdim,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return {values, indices};
}

// Registration function
void register_max_forward(pybind11::module& m) {
    m.def("max_forward", &max_forward,
          "Compute torch.max along a dimension (CUDA).\n"
          "Args:\n"
          "    x (Tensor): input CUDA float32 tensor\n"
          "    dim (int): dimension to reduce\n"
          "    keepdim (bool): whether to keep the reduced dimension\n"
          "Returns:\n"
          "    torch.return_types.max: (values, indices)",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
}

// Auto-register
REGISTER_BINDING(max_forward, register_max_forward);
