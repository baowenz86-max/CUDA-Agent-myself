#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void diagonal_scatter_launcher(
    const float* input,
    const float* src,
    float* output,
    int B,
    int H,
    int W,
    int offset,
    int dim1,
    int dim2,
    cudaStream_t stream
);

static inline int64_t compute_diag_length(int64_t H, int64_t W, int64_t offset) {
    if (offset >= 0) {
        return std::min(H, W - offset);
    } else {
        return std::min(H + offset, W);
    }
}

torch::Tensor diagonal_scatter(
    torch::Tensor input,
    torch::Tensor src,
    int64_t offset,
    int64_t dim1,
    int64_t dim2
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, H, W)");
    TORCH_CHECK(src.dim() == 2, "src must be 2D (B, L)");

    int64_t B = input.size(0);
    int64_t H = input.size(1);
    int64_t W = input.size(2);
    TORCH_CHECK(src.size(0) == B, "src batch size must match input batch size");

    // Contract: dim1=1, dim2=2 for this task, but we'll check and handle generically for contiguous case
    TORCH_CHECK(dim1 == 1 && dim2 == 2, "only dim1=1, dim2=2 is supported");

    // Compute expected diagonal length
    int64_t diag_len = compute_diag_length(H, W, offset);
    TORCH_CHECK(diag_len > 0, "offset results in empty diagonal");
    TORCH_CHECK(src.size(1) == diag_len, "src second dimension must equal diagonal length");

    // Ensure contiguous memory for kernel launch
    auto input_contig = input.contiguous();
    auto src_contig = src.contiguous();

    // Allocate output tensor (contiguous, same as input shape)
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    diagonal_scatter_launcher(
        input_contig.data_ptr<float>(),
        src_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(offset),
        static_cast<int>(dim1),
        static_cast<int>(dim2),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_diagonal_scatter(pybind11::module& m) {
    m.def("diagonal_scatter", &diagonal_scatter,
          "Diagonal scatter along dims",
          py::arg("input"),
          py::arg("src"),
          py::arg("offset"),
          py::arg("dim1"),
          py::arg("dim2"));
}

REGISTER_BINDING(diagonal_scatter, register_diagonal_scatter);
