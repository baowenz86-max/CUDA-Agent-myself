#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t total_indices,
    int64_t embedding_dim,
    cudaStream_t stream);

// PyTorch wrapper for embedding lookup
torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Validate device
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    // Validate dtypes
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    // Ensure weight is contiguous
    auto weight_contig = weight.contiguous();

    // Ensure indices is contiguous; if not, make contiguous
    auto indices_contig = indices.contiguous();

    // Get dimensions
    int64_t total_indices = indices_contig.numel();
    int64_t embedding_dim = weight_contig.size(1);

    // Allocate output tensor with shape indices.shape + [embedding_dim]
    std::vector<int64_t> output_shape(indices_contig.sizes().begin(), indices_contig.sizes().end());
    output_shape.push_back(embedding_dim);
    auto output = torch::empty(output_shape, indices_contig.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_forward_launcher(
        indices_contig.data_ptr<int64_t>(),
        weight_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_indices,
        embedding_dim,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_embedding_forward(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding forward pass",
          py::arg("indices"), py::arg("weight"));
}

REGISTER_BINDING(embedding_forward, register_embedding_forward);
