#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void embedding_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream);

extern "C" void log_softmax_launcher(
    const float* input,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    int dim,
    cudaStream_t stream);

// Embedding forward wrapper
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Parse dimensions
    int64_t batch_size = input.size(0);
    int64_t seq_length = input.size(1);
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");
    TORCH_CHECK(batch_size > 0 && seq_length > 0, "input dimensions must be positive");

    // Allocate output
    auto output = torch::empty({batch_size, seq_length, embedding_dim}, input.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(seq_length),
        static_cast<int>(embedding_dim),
        static_cast<int>(num_embeddings),
        stream);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Log softmax forward wrapper
torch::Tensor log_softmax_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    // Parse dimensions (assume input shape [batch, seq, embedding_dim])
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    int64_t batch = input.size(0);
    int64_t seq = input.size(1);
    int64_t embedding_dim = input.size(2);

    TORCH_CHECK(batch > 0 && seq > 0 && embedding_dim > 0, "dimensions must be positive");

    // Allocate output same shape as input
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    log_softmax_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(seq),
        static_cast<int>(embedding_dim),
        static_cast<int>(dim),
        stream);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward",
          py::arg("input"), py::arg("weight"));
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax forward",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
