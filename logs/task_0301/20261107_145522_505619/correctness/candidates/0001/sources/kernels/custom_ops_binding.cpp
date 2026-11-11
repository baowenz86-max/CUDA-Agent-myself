#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void batched_matmul_cuda(
    const float* a,
    const float* b,
    float* out,
    int B,
    int M,
    int N,
    int K,
    bool transpose_a,
    bool transpose_b,
    cudaStream_t stream
);

static torch::Tensor batched_matmul(
    torch::Tensor a,
    torch::Tensor b,
    bool transpose_a,
    bool transpose_b
) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    TORCH_CHECK(a.dim() == 3, "a must be 3D [B, M, K]");
    TORCH_CHECK(b.dim() == 3, "b must be 3D [B, K, N]");

    int B = a.size(0);
    int M = a.size(1);
    int K_a = a.size(2);
    int B_b = b.size(0);
    int K_b = b.size(1);
    int N = b.size(2);

    TORCH_CHECK(B == B_b, "batch sizes must match");
    int K_effective = transpose_a ? M : K_a;
    int K_expected = transpose_b ? N : K_b;
    TORCH_CHECK(K_effective == K_expected, "inner dimensions must match after considering transposes");

    int M_out = transpose_a ? K_a : M;
    int N_out = transpose_b ? K_b : N;

    auto out = torch::empty({B, M_out, N_out}, a.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batched_matmul_cuda(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        M,
        N,
        transpose_a ? K_a : K_b,  // K dimension used by launcher (original K)
        transpose_a,
        transpose_b,
        stream
    );

    return out;
}

void register_batched_matmul(pybind11::module& m) {
    m.def(
        "batched_matmul",
        &batched_matmul,
        "Batched matrix multiplication using cuBLAS",
        pybind11::arg("a"),
        pybind11::arg("b"),
        pybind11::arg("transpose_a"),
        pybind11::arg("transpose_b")
    );
}

REGISTER_BINDING(batched_matmul, register_batched_matmul);
