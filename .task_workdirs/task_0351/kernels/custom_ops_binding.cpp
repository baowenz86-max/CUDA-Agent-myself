#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t length,
    int64_t embedding_dim,
    cudaStream_t stream);

extern "C" void scalar_mul_forward_launcher(
    const float* input,
    float* output,
    float scalar,
    int64_t numel,
    cudaStream_t stream);

// Embedding forward wrapper
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    // Check input tensor
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Check weight tensor
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Get dimensions
    auto input_sizes = input.sizes(); // vector<int64_t>
    int64_t num_input_elements = input.numel();
    int64_t embedding_dim = weight.size(1);

    // Allocate output: shape = input_sizes + [embedding_dim]
    std::vector<int64_t> output_sizes(input_sizes.begin(), input_sizes.end());
    output_sizes.push_back(embedding_dim);
    auto output = torch::empty(output_sizes, input.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // The launcher expects batch and length as separate dimensions.
    // For arbitrary input shape, we pass num_input_elements as batch and length=1,
    // but the kernel uses total_input_elements = batch * length.
    // To ensure correct indexing, we can pass batch = num_input_elements, length = 1.
    // The kernel output is then total_input_elements * embedding_dim.
    // Note: For shape [B, L], this works as expected because the output is flattened
    // and contiguous. The kernel's indexing uses (input_offset * embedding_dim + dim_offset),
    // so it works regardless of input rank, as long as total output elements match.
    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        num_input_elements, // batch
        1,                 // length
        embedding_dim,
        stream
    );

    return output;
}

// Scalar multiplication wrapper
torch::Tensor scalar_mul_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    scalar_mul_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scalar),
        input.numel(),
        stream
    );

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding lookup forward",
          py::arg("input"), py::arg("weight"));
    m.def("scalar_mul_forward", &scalar_mul_forward, "Scalar multiplication forward",
          py::arg("input"), py::arg("scalar"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
