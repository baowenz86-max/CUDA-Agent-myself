#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include "../binding_registry.h"

// Declaration of the launcher from triu_forward.cu
extern "C" void triu_forward_launcher(
    void* output,
    const void* input,
    int64_t num_elements,
    int64_t rows,
    int64_t cols,
    int64_t diagonal,
    int64_t dtype,
    cudaStream_t stream);

// Helper to map torch dtype to dtype code used in launcher
static int64_t get_dtype_code(const at::Tensor& t) {
    auto dtype = t.scalar_type();
    if (dtype == at::kFloat) return 0;
    if (dtype == at::kDouble) return 1;
    if (dtype == at::kHalf) return 2;
    if (dtype == at::kByte) return 3;
    if (dtype == at::kInt) return 4;
    if (dtype == at::kLong) return 5;
    if (dtype == at::kBool) return 6;
    TORCH_CHECK(false, "Unsupported dtype for triu_forward");
}

// PyTorch wrapper
at::Tensor triu_forward(at::Tensor input, int64_t diagonal = 0) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    // For triu, we treat the last two dimensions as matrix dimensions.
    // The input shape is (N, C, H, W) -> N*C matrices of size H x W.
    int64_t rows = input.size(-2);
    int64_t cols = input.size(-1);
    int64_t num_matrices = 1;
    for (int64_t i = 0; i < input.dim() - 2; ++i) {
        num_matrices *= input.size(i);
    }
    int64_t num_elements = input.numel();

    // Allocate output tensor with same shape as input
    auto output = at::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get dtype code
    int64_t dtype_code = get_dtype_code(input);

    // Call launcher
    triu_forward_launcher(
        output.data_ptr(),
        input.data_ptr(),
        num_elements,
        rows,
        cols,
        diagonal,
        dtype_code,
        stream);

    return output;
}

// Registration function
void register_triu_forward(pybind11::module& m) {
    m.def("triu_forward", &triu_forward,
          "Upper triangular mask operation",
          pybind11::arg("input"), pybind11::arg("diagonal") = 0);
}

// Auto-register
REGISTER_BINDING(triu_forward, register_triu_forward);
