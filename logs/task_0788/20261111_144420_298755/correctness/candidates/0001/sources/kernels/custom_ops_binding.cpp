#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare the launcher from custom_ops.cu
extern "C" void custom_softmax_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

// Wrapper function for Python binding
torch::Tensor custom_softmax(torch::Tensor x, int64_t dim) {
    // Check tensor properties
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Ensure contiguous (though input likely is)
    x = x.contiguous();

    // Allocate output tensor with same shape and options
    auto output = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_softmax_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        N, C, H, W,
        stream
    );

    // Check for kernel launch errors (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_softmax(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax,
          "Custom softmax along dimension 1 of a 4D tensor",
          py::arg("x"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_softmax, register_custom_softmax);
