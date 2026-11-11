#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t n, cudaStream_t stream);

// PyTorch wrapper for element-wise ReLU
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() >= 0, "Input must have non-negative number of elements");

    auto output = torch::empty_like(x);
    if (x.numel() == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_relu(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "Element-wise ReLU", py::arg("x"));
}

REGISTER_BINDING(relu, register_relu);
