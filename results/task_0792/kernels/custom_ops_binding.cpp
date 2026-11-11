#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);
extern "C" void abs_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);

// PyTorch wrapper for ReLU forward
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_forward_launcher failed: ", cudaGetErrorString(err));
    return output;
}

// PyTorch wrapper for absolute value forward
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_forward_launcher failed: ", cudaGetErrorString(err));
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
    m.def("abs_forward", &abs_forward, "Absolute value forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
