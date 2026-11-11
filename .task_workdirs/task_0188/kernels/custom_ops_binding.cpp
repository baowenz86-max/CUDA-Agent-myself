#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void cummin_forward_launcher(
    const float* x,
    float* out,
    int N,
    int D,
    int64_t sx0,
    int64_t sx1,
    int64_t sy0,
    int64_t sy1,
    cudaStream_t stream);

// PyTorch wrapper for cumulative minimum along dimension 0
torch::Tensor cummin_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions (N, D, ...)");

    // For this specific task, input is [N, D] with N=10000, D=128. We support 2D only.
    TORCH_CHECK(x.dim() == 2, "cummin_forward currently supports only 2D input [N, D]");

    const int64_t N = x.size(0);
    const int64_t D = x.size(1);
    TORCH_CHECK(N > 0 && D > 0, "Input dimensions must be positive");

    // Input strides in elements
    const int64_t sx0 = x.stride(0);
    const int64_t sx1 = x.stride(1);

    // Allocate output using empty_like: same shape, stride, dtype, device as input
    auto out = torch::empty_like(x);

    // Output strides in elements (should match input, but we pass them explicitly)
    const int64_t sy0 = out.stride(0);
    const int64_t sy1 = out.stride(1);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cummin_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(D),
        sx0,
        sx1,
        sy0,
        sy1,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_cummin(pybind11::module& m) {
    m.def("cummin_forward", &cummin_forward,
          "Cumulative minimum along dimension 0 (values only)",
          pybind11::arg("x"));
}

REGISTER_BINDING(cummin, register_cummin);
