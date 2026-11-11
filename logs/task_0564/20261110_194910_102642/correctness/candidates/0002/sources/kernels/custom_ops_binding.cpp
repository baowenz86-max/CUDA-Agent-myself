// Binding file for max_dim1 and log1p custom ops
// All CUDA kernels are defined in custom_ops.cu; this file only handles PyTorch tensor wrapping and pybind registration.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void max_dim1_launcher(const float* input,
                                  float* output,
                                  int B, int C, int H, int W,
                                  cudaStream_t stream);

extern "C" void log1p_launcher(const float* input,
                               float* output,
                               int N,
                               cudaStream_t stream);

// PyTorch wrapper for max over dimension 1
// Input: x [B, C, H, W] float32 CUDA contiguous
// Output: [B, H, W] float32 CUDA tensor
torch::Tensor max_dim1(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional [B,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor [B, H, W]
    auto output = torch::empty({B, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    max_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for log1p elementwise (renamed to avoid conflict with std::log1p)
torch::Tensor log1p_forward(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Allocate output tensor with same shape as input
    auto output = torch::empty_like(x);

    int N = x.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    log1p_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log1p kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim1", &max_dim1, "Custom max over dimension 1", py::arg("x"));
    m.def("log1p", &log1p_forward, "Custom log1p elementwise", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);