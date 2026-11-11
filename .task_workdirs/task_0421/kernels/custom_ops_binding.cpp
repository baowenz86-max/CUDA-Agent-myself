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
    int batch_size,
    int sequence_length,
    int embedding_dim,
    int64_t num_embeddings,
    cudaStream_t stream);

extern "C" void bitwise_xor_int64_launcher(
    const float* x,
    const float* y,
    float* output,
    int batch_size,
    int sequence_length,
    int embedding_dim,
    cudaStream_t stream);

// embedding_forward tensor wrapper
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    // Check device
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    // Check contiguity
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    // Check dimensions
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, sequence_length]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");

    int64_t batch_size = input.size(0);
    int64_t sequence_length = input.size(1);
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    TORCH_CHECK(batch_size > 0 && sequence_length > 0 && num_embeddings > 0 && embedding_dim > 0,
                "dimensions must be positive");

    // Allocate output tensor [batch_size, sequence_length, embedding_dim]
    auto output = torch::empty(
        {batch_size, sequence_length, embedding_dim},
        input.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(sequence_length),
        static_cast<int>(embedding_dim),
        num_embeddings,
        stream
    );

    return output;
}

// bitwise_xor_int64 tensor wrapper
torch::Tensor bitwise_xor_int64(torch::Tensor x, torch::Tensor y) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    // Check dimensions
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch_size, sequence_length, embedding_dim]");
    TORCH_CHECK(y.dim() == 2, "y must be 2D [sequence_length, embedding_dim]");

    int64_t batch_size = x.size(0);
    int64_t sequence_length = x.size(1);
    int64_t embedding_dim = x.size(2);

    TORCH_CHECK(y.size(0) == sequence_length && y.size(1) == embedding_dim,
                "y shape must match [sequence_length, embedding_dim]");
    TORCH_CHECK(batch_size > 0 && sequence_length > 0 && embedding_dim > 0,
                "dimensions must be positive");

    // Allocate output tensor [batch_size, sequence_length, embedding_dim]
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    bitwise_xor_int64_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(sequence_length),
        static_cast<int>(embedding_dim),
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding lookup forward",
          py::arg("input"), py::arg("weight"));
    m.def("bitwise_xor_int64", &bitwise_xor_int64,
          "Bitwise XOR int64 forward",
          py::arg("x"), py::arg("y"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
