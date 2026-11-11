#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_reduce_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t dim_size,
    int64_t width,
    cudaStream_t stream
);

extern "C" void softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t width,
    cudaStream_t stream
);

// Wrapper for max_reduce_forward: reduces max along dimension 1 of a 3D tensor
torch::Tensor max_reduce_forward(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for max_reduce_forward");

    int64_t batch_size = x.size(0);
    int64_t dim_size = x.size(1);
    int64_t width = x.size(2);

    // Allocate output tensor [batch_size, width]
    auto output = torch::empty({batch_size, width}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    max_reduce_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size,
        width,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_reduce_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for softmax_forward: softmax along dimension 1 of a 2D tensor
torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for softmax_forward");

    int64_t batch_size = x.size(0);
    int64_t width = x.size(1);

    // Allocate output tensor same shape
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        width,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("max_reduce_forward", &max_reduce_forward,
          "Max reduce along dimension 1",
          py::arg("x"), py::arg("dim"));
    m.def("softmax_forward", &softmax_forward,
          "Softmax along dimension 1",
          py::arg("x"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
