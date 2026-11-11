#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_bag_forward_launcher(
    const int64_t* input_cpu,
    const int64_t* offsets_cpu,
    const float* weight_cpu,
    int64_t num_bags,
    int64_t embedding_dim,
    int64_t num_embeddings,
    float* output_cuda,
    cudaStream_t stream);

torch::Tensor embedding_bag_forward(torch::Tensor input, torch::Tensor offsets, torch::Tensor weight) {
    // Move tensors to CPU if they are not already on CPU
    auto input_cpu = input.is_cpu() ? input : input.cpu();
    auto offsets_cpu = offsets.is_cpu() ? offsets : offsets.cpu();
    auto weight_cpu = weight.is_cpu() ? weight : weight.cpu();

    // Validate dtypes
    TORCH_CHECK(input_cpu.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(offsets_cpu.dtype() == torch::kInt64, "offsets must be int64");
    TORCH_CHECK(weight_cpu.dtype() == torch::kFloat32, "weight must be float32");

    // Validate contiguity
    TORCH_CHECK(input_cpu.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(offsets_cpu.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(weight_cpu.is_contiguous(), "weight must be contiguous");

    // Validate dimensions
    TORCH_CHECK(input_cpu.dim() == 1, "input must be 1D");
    TORCH_CHECK(offsets_cpu.dim() == 1, "offsets must be 1D");
    TORCH_CHECK(weight_cpu.dim() == 2, "weight must be 2D");

    int64_t num_bags = offsets_cpu.numel() - 1;
    TORCH_CHECK(num_bags > 0, "offsets must have at least 2 elements");
    int64_t embedding_dim = weight_cpu.size(1);
    int64_t num_embeddings = weight_cpu.size(0);

    // Validate offsets: first element must be 0, non-decreasing, last <= input.numel()
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
    TORCH_CHECK(offsets_acc[0] == 0, "offsets[0] must be 0");
    for (int64_t i = 0; i < num_bags; ++i) {
        TORCH_CHECK(offsets_acc[i] <= offsets_acc[i + 1], "offsets must be non-decreasing");
    }
    TORCH_CHECK(offsets_acc[num_bags] <= input_cpu.numel(), "offsets last element must be <= input length");

    // Determine output device: use current CUDA device (since execution_device is cuda)
    auto output = torch::empty({num_bags, embedding_dim},
                               torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_bag_forward_launcher(
        input_cpu.data_ptr<int64_t>(),
        offsets_cpu.data_ptr<int64_t>(),
        weight_cpu.data_ptr<float>(),
        num_bags,
        embedding_dim,
        num_embeddings,
        output.data_ptr<float>(),
        stream
    );

    return output;
}

void register_embedding_bag(pybind11::module& m) {
    m.def("embedding_bag_forward", &embedding_bag_forward,
          "EmbeddingBag forward (sum mode)",
          py::arg("input"), py::arg("offsets"), py::arg("weight"));
}

REGISTER_BINDING(embedding_bag, register_embedding_bag);