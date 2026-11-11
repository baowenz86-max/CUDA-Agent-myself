#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the launcher from custom_ops.cu
extern "C" void logical_or_round_launcher(
    const bool* x1,
    const bool* x2,
    float* out,
    int64_t numel,
    cudaStream_t stream);

// PyTorch tensor wrapper
static torch::Tensor logical_or_round_forward(torch::Tensor x1, torch::Tensor x2) {
    // Input validation
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kBool, "x1 must be bool");
    TORCH_CHECK(x2.dtype() == torch::kBool, "x2 must be bool");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");

    // Ensure contiguous (the kernel assumes linear contiguous layout)
    auto x1_contig = x1.is_contiguous() ? x1 : x1.contiguous();
    auto x2_contig = x2.is_contiguous() ? x2 : x2.contiguous();

    // Allocate output float32 tensor with same shape
    auto out = torch::empty_like(x1_contig, torch::TensorOptions().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    int64_t numel = x1_contig.numel();
    logical_or_round_launcher(
        x1_contig.data_ptr<bool>(),
        x2_contig.data_ptr<bool>(),
        out.data_ptr<float>(),
        numel,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
static void register_logical_or_round(pybind11::module& m) {
    m.def("logical_or_round", &logical_or_round_forward,
          "Compute logical_or of two bool tensors and return float32 rounded result",
          py::arg("x1"), py::arg("x2"));
}

// Auto-register
REGISTER_BINDING(logical_or_round, register_logical_or_round);
