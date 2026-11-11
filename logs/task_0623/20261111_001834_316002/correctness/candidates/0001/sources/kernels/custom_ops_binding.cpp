#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_lookup_gelu_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t num_indices,
    int64_t embedding_dim,
    cudaStream_t stream
);

// PyTorch binding wrapper
torch::Tensor embedding_lookup_gelu(torch::Tensor indices, torch::Tensor weight) {
    // Ensure indices and weight are on CUDA
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");

    // Move indices to CUDA if needed
    if (!indices.is_cuda()) {
        indices = indices.to(torch::kCUDA);
    }

    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    int64_t batch_size = indices.size(0);
    int64_t seq_len = indices.size(1);
    int64_t num_indices = batch_size * seq_len;

    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    // Optionally check indices range (debug only). Not necessary for performance.
    // TORCH_CHECK(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < num_embeddings,
    //             "indices out of range");

    // Allocate output tensor on CUDA
    auto output = torch::empty({batch_size, seq_len, embedding_dim}, weight.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel via launcher
    embedding_lookup_gelu_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        num_indices,
        embedding_dim,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_embedding_lookup_gelu(pybind11::module& m) {
    m.def("embedding_lookup_gelu", &embedding_lookup_gelu,
          "Fused embedding lookup followed by GELU activation",
          py::arg("indices"), py::arg("weight"));
}

// Auto-register
REGISTER_BINDING(embedding_lookup_gelu, register_embedding_lookup_gelu);
