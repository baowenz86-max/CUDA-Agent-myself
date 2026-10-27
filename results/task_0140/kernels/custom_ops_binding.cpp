#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardsigmoid_forward_launcher(const float* input,
                                             float* output,
                                             int64_t numel,
                                             cudaStream_t stream);

extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

// Wrapper for hardsigmoid activation
torch::Tensor hardsigmoid_forward(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "hardsigmoid_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardsigmoid_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "hardsigmoid_forward: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardsigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardsigmoid_forward: CUDA kernel launch failed with error ",
                cudaGetErrorString(err));

    return output;
}

// Wrapper for ReLU activation
torch::Tensor relu_forward(torch::Tensor y) {
    // Input checks
    TORCH_CHECK(y.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    TORCH_CHECK(y.is_contiguous(), "relu_forward: input must be contiguous");

    auto output = torch::empty_like(y);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        y.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_forward: CUDA kernel launch failed with error ",
                cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("hardsigmoid_forward", &hardsigmoid_forward, "HardSigmoid activation forward",
          py::arg("x"));
    m.def("relu_forward", &relu_forward, "ReLU activation forward",
          py::arg("y"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
