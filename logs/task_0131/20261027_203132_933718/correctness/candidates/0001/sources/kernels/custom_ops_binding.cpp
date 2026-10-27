#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_relu_launcher(const float* input, float* output, long long numel, cudaStream_t stream);
extern "C" void custom_gelu_launcher(const float* input, float* output, long long numel, cudaStream_t stream);

// Helper to validate input tensor
static void check_input(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(x.numel() > 0, name, " must have at least one element");
}

// ReLU forward
static torch::Tensor custom_relu_forward(torch::Tensor x) {
    check_input(x, "custom_relu input");
    
    // Ensure contiguous for launcher
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_relu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_relu kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// GELU forward
static torch::Tensor custom_gelu_forward(torch::Tensor x) {
    check_input(x, "custom_gelu input");
    
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_gelu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_gelu kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Register both functions
static void register_custom_ops(pybind11::module& m) {
    m.def("custom_relu", &custom_relu_forward, "ReLU forward (CUDA)",
          pybind11::arg("x"));
    m.def("custom_gelu", &custom_gelu_forward, "GELU forward (CUDA)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
