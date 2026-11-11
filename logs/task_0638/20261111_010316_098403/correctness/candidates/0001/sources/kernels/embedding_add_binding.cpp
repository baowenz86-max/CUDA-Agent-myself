// kernels/embedding_add_binding.cpp
// PyTorch binding for embedding_add_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_add_launcher(
    const int64_t* x1,
    const int64_t* x2,
    const float* weight1,
    const float* weight2,
    float* output,
    int64_t num_embeddings_1,
    int64_t num_embeddings_2,
    int64_t embedding_dim,
    int64_t batch,
    int64_t seq_len,
    int64_t stride_x1_batch,
    int64_t stride_x1_seq,
    int64_t stride_x2_batch,
    int64_t stride_x2_seq,
    cudaStream_t stream);

torch::Tensor embedding_add_forward(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor weight1,
    torch::Tensor weight2,
    int64_t num_embeddings_1,
    int64_t num_embeddings_2,
    int64_t embedding_dim)
{
    // Check CUDA device
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(weight1.is_cuda(), "weight1 must be a CUDA tensor");
    TORCH_CHECK(weight2.is_cuda(), "weight2 must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(x1.dtype() == torch::kInt64, "x1 must be int64");
    TORCH_CHECK(x2.dtype() == torch::kInt64, "x2 must be int64");
    TORCH_CHECK(weight1.dtype() == torch::kFloat32, "weight1 must be float32");
    TORCH_CHECK(weight2.dtype() == torch::kFloat32, "weight2 must be float32");

    // Check shapes
    TORCH_CHECK(x1.dim() == 2, "x1 must be 2D");
    TORCH_CHECK(x2.dim() == 2, "x2 must be 2D");
    TORCH_CHECK(x1.size(0) == x2.size(0), "batch dimensions must match");
    TORCH_CHECK(x1.size(1) == x2.size(1), "seq_len dimensions must match");
    int64_t batch = x1.size(0);
    int64_t seq_len = x1.size(1);
    TORCH_CHECK(weight1.dim() == 2, "weight1 must be 2D");
    TORCH_CHECK(weight2.dim() == 2, "weight2 must be 2D");
    TORCH_CHECK(weight1.size(0) == num_embeddings_1, "weight1 first dim mismatch");
    TORCH_CHECK(weight1.size(1) == embedding_dim, "weight1 second dim mismatch");
    TORCH_CHECK(weight2.size(0) == num_embeddings_2, "weight2 first dim mismatch");
    TORCH_CHECK(weight2.size(1) == embedding_dim, "weight2 second dim mismatch");

    // Compute strides (assume row-major; if not contiguous, strides are in elements)
    int64_t stride_x1_batch = x1.stride(0);
    int64_t stride_x1_seq = x1.stride(1);
    int64_t stride_x2_batch = x2.stride(0);
    int64_t stride_x2_seq = x2.stride(1);

    // Allocate output with same device and dtype
    auto output = torch::empty({batch, seq_len, embedding_dim}, x1.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_add_launcher(
        x1.data_ptr<int64_t>(),
        x2.data_ptr<int64_t>(),
        weight1.data_ptr<float>(),
        weight2.data_ptr<float>(),
        output.data_ptr<float>(),
        num_embeddings_1,
        num_embeddings_2,
        embedding_dim,
        batch,
        seq_len,
        stride_x1_batch,
        stride_x1_seq,
        stride_x2_batch,
        stride_x2_seq,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_embedding_add_forward(pybind11::module& m) {
    m.def("embedding_add_forward", &embedding_add_forward,
          "Compute embedding1(x1) + embedding2(x2)",
          py::arg("x1"),
          py::arg("x2"),
          py::arg("weight1"),
          py::arg("weight2"),
          py::arg("num_embeddings_1"),
          py::arg("num_embeddings_2"),
          py::arg("embedding_dim"));
}

REGISTER_BINDING(embedding_add_forward, register_embedding_add_forward);
