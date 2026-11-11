// Binding for custom operations: baddbmm_custom and equal_custom
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void baddbmm_custom_launcher(
    float* output,
    const float* input,
    const float* mat1,
    const float* mat2,
    int B, int M, int N, int K,
    long long int input_stride_0, long long int input_stride_1,
    long long int mat1_stride_0, long long int mat1_stride_1,
    long long int mat2_stride_0, long long int mat2_stride_1,
    cudaStream_t stream);

extern "C" bool equal_custom_launcher(
    const float* lhs,
    const float* rhs,
    long long numel,
    cudaStream_t stream);

// Wrapper for baddbmm_custom: out = input + mat1 @ mat2
torch::Tensor baddbmm_custom(torch::Tensor input, torch::Tensor mat1, torch::Tensor mat2) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK(mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mat1.dtype() == torch::kFloat32, "mat1 must be float32");
    TORCH_CHECK(mat2.dtype() == torch::kFloat32, "mat2 must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be 3D");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D");

    // Ensure contiguous for cuBLAS, supporting arbitrary strides by making a contiguous copy
    auto input_contig = input.contiguous();
    auto mat1_contig = mat1.contiguous();
    auto mat2_contig = mat2.contiguous();

    int64_t B = input_contig.size(0);
    int64_t M = input_contig.size(1);
    int64_t N = input_contig.size(2);
    TORCH_CHECK(mat1_contig.size(0) == B && mat1_contig.size(1) == M, "mat1 batch and rows must match input");
    int64_t K = mat1_contig.size(2);
    TORCH_CHECK(mat2_contig.size(0) == B && mat2_contig.size(1) == K && mat2_contig.size(2) == N, "mat2 dimensions mismatch");

    // Output allocation using empty_like (only allocation, no torch compute)
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Since we made contiguous, strides are now standard
    long long int input_stride_0 = M * N;
    long long int input_stride_1 = N;
    long long int mat1_stride_0 = M * K;
    long long int mat1_stride_1 = K;
    long long int mat2_stride_0 = K * N;
    long long int mat2_stride_1 = N;

    baddbmm_custom_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        mat1_contig.data_ptr<float>(),
        mat2_contig.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        input_stride_0, input_stride_1,
        mat1_stride_0, mat1_stride_1,
        mat2_stride_0, mat2_stride_1,
        stream
    );

    return output;
}

// Wrapper for equal_custom: compare two tensors elementwise, return bool
bool equal_custom(torch::Tensor lhs, torch::Tensor rhs) {
    // Input validation
    TORCH_CHECK(lhs.is_cuda(), "lhs must be a CUDA tensor");
    TORCH_CHECK(rhs.is_cuda(), "rhs must be a CUDA tensor");
    TORCH_CHECK(lhs.dtype() == torch::kFloat32, "lhs must be float32");
    TORCH_CHECK(rhs.dtype() == torch::kFloat32, "rhs must be float32");

    // Shape check: torch.equal returns False if shapes differ
    if (lhs.sizes() != rhs.sizes()) {
        return false;
    }

    // Ensure contiguous for linear access in kernel
    auto lhs_contig = lhs.contiguous();
    auto rhs_contig = rhs.contiguous();

    // Get number of elements
    long long numel = static_cast<long long>(lhs_contig.numel());
    if (numel == 0) {
        return true; // empty tensors are equal
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (returns bool after synchronizing internally)
    return equal_custom_launcher(
        lhs_contig.data_ptr<float>(),
        rhs_contig.data_ptr<float>(),
        numel,
        stream
    );
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("baddbmm_custom", &baddbmm_custom, "Batched matrix multiply-add: out = input + mat1 @ mat2",
          py::arg("input"), py::arg("mat1"), py::arg("mat2"));
    m.def("equal_custom", &equal_custom, "Elementwise equality of two tensors",
          py::arg("lhs"), py::arg("rhs"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
