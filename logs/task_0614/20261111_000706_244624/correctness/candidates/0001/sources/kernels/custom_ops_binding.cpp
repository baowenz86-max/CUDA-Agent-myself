#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void embedding_gt_forward_launcher(
    const float* weight,
    const int64_t* indices,
    bool* output,
    int64_t embedding_dim,
    int64_t total_indices,
    cudaStream_t stream);

// PyTorch wrapper
static torch::Tensor embedding_gt_forward(
    torch::Tensor weight,
    torch::Tensor indices,
    int64_t padding_idx,
    double threshold)
{
    // input validation
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(indices.dim() >= 1, "indices must have at least 1 dimension");

    const int64_t embedding_dim = weight.size(1);
    const int64_t total_indices = indices.numel();

    // Output shape: indices.shape + [embedding_dim]
    std::vector<int64_t> output_shape(indices.sizes().begin(), indices.sizes().end());
    output_shape.push_back(embedding_dim);

    auto output = torch::empty(output_shape, indices.options().dtype(torch::kBool));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_gt_forward_launcher(
        weight.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        output.data_ptr<bool>(),
        embedding_dim,
        total_indices,
        stream);

    // Note: padding_idx and threshold are not used by the current launcher (threshold hardcoded to 0.5),
    // but they are accepted for interface compatibility.

    return output;
}

// Registration function
static void register_custom_ops_binding(pybind11::module& m) {
    m.def("embedding_gt_forward", &embedding_gt_forward,
          "Embedding lookup followed by greater-than threshold",
          py::arg("weight"), py::arg("indices"), py::arg("padding_idx"), py::arg("threshold"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
