#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_bag_sum_forward_launcher(
    float* output,
    const int64_t* input_cpu,
    const int64_t* offsets_cpu,
    const float* weight_cpu,
    int total_indices,
    int num_bags,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream);

torch::Tensor embedding_bag_sum_forward(
    torch::Tensor input,
    torch::Tensor offsets,
    torch::Tensor weight) {
    // Input validation
    TORCH_CHECK(input.is_cpu(), "input must be a CPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(input.dim() == 1, "input must be 1D");

    TORCH_CHECK(offsets.is_cpu(), "offsets must be a CPU tensor");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be int64");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");

    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    const int64_t num_bags_64 = offsets.numel() - 1;
    TORCH_CHECK(num_bags_64 > 0, "offsets must have at least 2 elements");
    TORCH_CHECK(num_bags_64 <= INT_MAX, "num_bags exceeds int range");
    const int num_bags = static_cast<int>(num_bags_64);

    const int64_t embedding_dim_64 = weight.size(1);
    TORCH_CHECK(embedding_dim_64 > 0 && embedding_dim_64 <= 1024, "embedding_dim must be in (0, 1024]");
    const int embedding_dim = static_cast<int>(embedding_dim_64);

    const int64_t num_embeddings_64 = weight.size(0);
    TORCH_CHECK(num_embeddings_64 > 0 && num_embeddings_64 <= INT_MAX, "num_embeddings exceeds int range");
    const int num_embeddings = static_cast<int>(num_embeddings_64);

    const int64_t total_indices_64 = input.numel();
    TORCH_CHECK(total_indices_64 <= INT_MAX, "total indices exceed int range");
    const int total_indices = static_cast<int>(total_indices_64);

    // Check that all indices are within range? Not required by contract, but can do basic check.
    // Skip for performance; launcher kernel handles if valid.

    // Allocate output on CUDA
    auto output = torch::empty({num_bags, embedding_dim}, weight.options().device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_bag_sum_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        total_indices,
        num_bags,
        embedding_dim,
        num_embeddings,
        stream);

    return output;
}

void register_embedding_bag_sum_forward(pybind11::module& m) {
    m.def("embedding_bag_sum_forward", &embedding_bag_sum_forward,
          "EmbeddingBag sum forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("offsets"),
          pybind11::arg("weight"));
}

REGISTER_BINDING(embedding_bag_sum_forward, register_embedding_bag_sum_forward);
