// binding for fused einsum + bmm
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void einsum_bmm_fused_launcher(const float* x, float* z, int B, int N, cudaStream_t stream);

// PyTorch wrapper
torch::Tensor einsum_bmm_fused(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, N, N)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t B = x.size(0);
    int64_t N = x.size(1);
    TORCH_CHECK(N > 0, "N must be positive");
    TORCH_CHECK(x.size(2) == N, "x must be square in last two dimensions");

    auto z = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    einsum_bmm_fused_launcher(x.data_ptr<float>(), z.data_ptr<float>(), static_cast<int>(B), static_cast<int>(N), stream);

    return z;
}

// Registration
void register_einsum_bmm_fused(pybind11::module& m) {
    m.def("einsum_bmm_fused", &einsum_bmm_fused, "Fused einsum('bij,bjk->bik') + bmm(y,y)", pybind11::arg("x"));
}

REGISTER_BINDING(einsum_bmm_fused, register_einsum_bmm_fused);
