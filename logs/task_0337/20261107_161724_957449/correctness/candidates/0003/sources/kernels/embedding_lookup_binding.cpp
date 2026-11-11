// Binding for embedding_lookup custom op
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_lookup_launcher(
    const float* weight_cpu,
    const int64_t* indices_cpu,
    float* output_gpu,
    int num_embeddings,
    int height,
    int width,
    int embedding_dim,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor embedding_lookup(torch::Tensor weight, torch::Tensor indices) {
    // Check inputs
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(indices.is_cpu(), "indices must be a CPU tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [height, width]");

    int num_embeddings = weight.size(0);
    int embedding_dim = weight.size(1);
    int height = indices.size(0);
    int width = indices.size(1);

    // Allocate output tensor on CUDA
    auto output = torch::empty({height, width, embedding_dim}, weight.options().device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointers
    embedding_lookup_launcher(
        weight.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_embeddings,
        height,
        width,
        embedding_dim,
        stream
    );

    return output;
}

// Registration function
void register_embedding_lookup(pybind11::module& m) {
    m.def(
        "embedding_lookup",
        &embedding_lookup,
        "Embedding lookup (CPU weight/indices -> CUDA output)",
        pybind11::arg("weight"),
        pybind11::arg("indices")
    );
}

// Auto-register
REGISTER_BINDING(embedding_lookup, register_embedding_lookup);
