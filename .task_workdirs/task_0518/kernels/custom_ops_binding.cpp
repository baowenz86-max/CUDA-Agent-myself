#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void softmin_forward_launcher(const float* input,
                                         float* output,
                                         int64_t num_rows,
                                         int64_t feat_dim,
                                         cudaStream_t stream);

// PyTorch wrapper for softmin forward
torch::Tensor softmin_forward(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3-dimensional [batch, seq_len, features]");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    // Contract specifies input shape [64, 70000, 10] and dim=1
    // We implement softmin along the last dimension (features=10) by permuting
    // or reshaping if necessary. However, since dim is passed but launcher only
    // handles softmin over last dim, we need to prepare accordingly.
    // For simplicity and as per contract (dim=1), we permute so that dim=1 becomes
    // last dimension, compute softmin, then permute back.
    // Since input is contiguous and shape is fixed, we can use view as [batch*seq_len, features]
    // only when dim=2 or dim=-1. For dim=1 we need to transpose.
    
    if (dim != 2 && dim != -1) {
        // Move target dim to last position
        auto perm = std::vector<int64_t>(x.dim());
        for (int64_t i = 0; i < x.dim(); ++i) perm[i] = i;
        perm.erase(perm.begin() + dim);
        perm.push_back(dim);
        x = x.permute(perm).contiguous();
    }

    int64_t feat_dim = x.size(-1);
    int64_t num_rows = x.numel() / feat_dim;
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    softmin_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        feat_dim,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // If we permuted, permute back
    if (dim != 2 && dim != -1) {
        // Reconstruct original permutation
        auto inv_perm = std::vector<int64_t>(x.dim());
        auto perm = std::vector<int64_t>(x.dim());
        for (int64_t i = 0; i < x.dim(); ++i) perm[i] = i;
        perm.erase(perm.begin() + dim);
        perm.push_back(dim);
        for (int64_t i = 0; i < x.dim(); ++i) {
            inv_perm[perm[i]] = i;
        }
        output = output.permute(inv_perm).contiguous();
    }

    return output;
}

// Registration function
void register_softmin_forward(pybind11::module& m) {
    m.def("softmin_forward", &softmin_forward,
          "Softmin forward kernel",
          py::arg("x"),
          py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(softmin_forward, register_softmin_forward);
