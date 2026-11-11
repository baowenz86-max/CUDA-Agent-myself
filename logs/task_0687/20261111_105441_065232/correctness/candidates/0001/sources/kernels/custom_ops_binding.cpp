#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim0_launcher(const float* input,
                                  float* values,
                                  int64_t* indices,
                                  int rows,
                                  int cols,
                                  cudaStream_t stream);

extern "C" void cumsum_dim0_launcher(const float* input,
                                     float* output,
                                     int rows,
                                     int cols,
                                     cudaStream_t stream);

// Wrapper for max_dim0: torch.max returns a named tuple (values, indices)
torch::return_types::max max_dim0(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported");

    int64_t rows = x.size(0);
    int64_t cols = x.size(1);

    auto values = torch::empty({cols}, x.options());
    auto indices = torch::empty({cols}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_dim0_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        rows,
        cols,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_dim0 kernel launch failed: ", cudaGetErrorString(err));

    // Construct the torch.return_types.max structure with values and indices
    torch::return_types::max result;
    result.values = values;
    result.indices = indices;
    return result;
}

// Wrapper for cumsum_dim0: returns a tensor
torch::Tensor cumsum_dim0(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported");

    int64_t rows = x.size(0);
    int64_t cols = x.size(1);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cumsum_dim0_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cumsum_dim0 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim0", &max_dim0,
          "Compute max along dim=0 for 2D tensor",
          py::arg("x"), py::arg("dim"));
    m.def("cumsum_dim0", &cumsum_dim0,
          "Compute cumulative sum along dim=0 for 2D tensor",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
