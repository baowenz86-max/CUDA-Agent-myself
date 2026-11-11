#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    cudaStream_t stream);

extern "C" void mul_scalar_launcher(
    const float* input,
    const float* scalar,
    float* output,
    int n,
    cudaStream_t stream);

// PyTorch wrapper for embedding_forward
torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Validate indices
    TORCH_CHECK(indices.is_cpu(), "indices must be a CPU tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch, seq]");

    // Validate weight
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");

    int64_t batch = indices.size(0);
    int64_t seq = indices.size(1);
    int64_t embedding_dim = weight.size(1);

    // Ensure indices are in valid range (optional but good)
    // TORCH_CHECK(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < weight.size(0), "indices out of range");
    // Skip range check for performance; contract assumes valid indices.

    // Allocate output on CUDA
    auto output = torch::empty({batch, seq, embedding_dim}, weight.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(seq),
        static_cast<int>(embedding_dim),
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for mul_scalar
torch::Tensor mul_scalar(torch::Tensor input, torch::Tensor scalar) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Validate scalar
    TORCH_CHECK(scalar.is_cuda(), "scalar must be a CUDA tensor");
    TORCH_CHECK(scalar.dtype() == torch::kFloat32, "scalar must be float32");
    TORCH_CHECK(scalar.numel() == 1, "scalar must have exactly one element");
    TORCH_CHECK(scalar.is_contiguous(), "scalar must be contiguous");

    // Allocate output same shape as input
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t n = input.numel();

    // Call launcher
    mul_scalar_launcher(
        input.data_ptr<float>(),
        scalar.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(n),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mul_scalar_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding forward (CUDA)",
          pybind11::arg("indices"),
          pybind11::arg("weight"));
    m.def("mul_scalar", &mul_scalar,
          "Multiply tensor by scalar (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
