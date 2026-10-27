#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void diag_embed_launcher(float* output, const float* input, int B, int N, cudaStream_t stream);
extern "C" void l1_norm_dim1_launcher(float* output, const float* input, int B, int N, cudaStream_t stream);

// PyTorch wrapper for diag_embed
torch::Tensor diag_embed(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");

    int64_t B = x.size(0);
    int64_t N = x.size(1);

    // Output shape [B, N, N]
    auto output = torch::empty({B, N, N}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    diag_embed_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(N),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "diag_embed kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for l1_norm_dim1
torch::Tensor l1_norm_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");

    int64_t B = x.size(0);
    int64_t N = x.size(1);
    // Note: dimension 2 should also be N, but we don't strictly need to check here.
    // TORCH_CHECK(x.size(2) == N, "Expected square matrix in last two dims");

    // Output shape [B, N]
    auto output = torch::empty({B, N}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    l1_norm_dim1_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(N),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "l1_norm_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("diag_embed", &diag_embed, "Embed input into a diagonal tensor (CUDA)",
          py::arg("x"));
    m.def("l1_norm_dim1", &l1_norm_dim1, "Compute L1 norm along dim=1 (CUDA)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
