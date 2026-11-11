// Binding for embedding_lookup custom op
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_lookup_launcher(
    const float* weight_ptr,
    bool weight_on_gpu,
    const int64_t* indices_ptr,
    bool indices_on_gpu,
    float* output_gpu,
    int num_embeddings,
    int height,
    int width,
    int embedding_dim,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor embedding_lookup(torch::Tensor weight, torch::Tensor indices) {
    // Check dtype and shape
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [height, width]");

    int num_embeddings = weight.size(0);
    int embedding_dim = weight.size(1);
    int height = indices.size(0);
    int width = indices.size(1);

    // Allocate output tensor on CUDA (always)
    auto output = torch::empty({height, width, embedding_dim}, 
                               weight.options().device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Determine if weight and indices are already on GPU
    bool weight_on_gpu = weight.is_cuda();
    bool indices_on_gpu = indices.is_cuda();

    // Ensure tensors are contiguous; if not, make contiguous copies (on their respective device)
    torch::Tensor weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    torch::Tensor indices_contig = indices.is_contiguous() ? indices : indices.contiguous();

    // Call launcher with raw pointers
    embedding_lookup_launcher(
        weight_contig.data_ptr<float>(),
        weight_on_gpu,
        indices_contig.data_ptr<int64_t>(),
        indices_on_gpu,
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
        "Embedding lookup (CPU/GPU weight and indices -> CUDA output)",
        pybind11::arg("weight"),
        pybind11::arg("indices")
    );
}

// Auto-register
REGISTER_BINDING(embedding_lookup, register_embedding_lookup);
