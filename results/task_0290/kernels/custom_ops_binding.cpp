#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void tril_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int64_t rows,
    int64_t cols,
    int64_t offset,
    cudaStream_t stream);

// PyTorch wrapper for tril_forward
torch::Tensor tril_forward(torch::Tensor x, int64_t offset) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");

    // Ensure contiguous for kernel's linear indexing (also correct for strided inputs)
    auto x_contig = x.contiguous();

    // Allocate output with same shape and contiguous layout
    auto output = torch::empty_like(x_contig);

    int64_t rows = x_contig.size(-2);
    int64_t cols = x_contig.size(-1);
    int64_t total_elements = x_contig.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch CUDA kernel
    tril_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        rows,
        cols,
        offset,
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tril_forward kernel launch failed");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("tril_forward", &tril_forward,
          "Lower triangular part of each matrix formed by the last two dimensions",
          pybind11::arg("x"),
          pybind11::arg("offset"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
