// Binding file for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax1d_launcher(const float* input, float* output, int B, int D, int C, cudaStream_t stream);
extern "C" void squeeze_launcher(const float* input, float* output, size_t numel, cudaStream_t stream);
extern "C" void add_scalar_launcher(const float* input, float* output, double value, size_t numel, cudaStream_t stream);

// Wrapper for softmax1d_forward
torch::Tensor softmax1d_forward(torch::Tensor x, int64_t dim) {
    // Check input
    TORCH_CHECK(x.is_cuda(), "softmax1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax1d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "softmax1d_forward: input must be 3D");
    TORCH_CHECK(dim == 1, "softmax1d_forward: only dim=1 supported");
    TORCH_CHECK(x.is_contiguous(), "softmax1d_forward: input must be contiguous");

    auto sizes = x.sizes();
    int B = static_cast<int>(sizes[0]);
    int D = static_cast<int>(sizes[1]);
    int C = static_cast<int>(sizes[2]);

    // Allocate output with same shape and options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    softmax1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D, C,
        stream
    );

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax1d_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for squeeze_forward
torch::Tensor squeeze_forward(torch::Tensor x) {
    // Check input
    TORCH_CHECK(x.is_cuda(), "squeeze_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "squeeze_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "squeeze_forward: input must be contiguous");

    size_t numel = x.numel();

    // Allocate output with same shape and options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    squeeze_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "squeeze_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for add_scalar_forward
torch::Tensor add_scalar_forward(torch::Tensor x, double value) {
    // Check input
    TORCH_CHECK(x.is_cuda(), "add_scalar_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "add_scalar_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "add_scalar_forward: input must be contiguous");

    size_t numel = x.numel();

    // Allocate output with same shape and options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    add_scalar_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        value,
        numel,
        stream
    );

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add_scalar_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("softmax1d_forward", &softmax1d_forward,
          py::arg("x"), py::arg("dim"));
    m.def("squeeze_forward", &squeeze_forward,
          py::arg("x"));
    m.def("add_scalar_forward", &add_scalar_forward,
          py::arg("x"), py::arg("value"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
