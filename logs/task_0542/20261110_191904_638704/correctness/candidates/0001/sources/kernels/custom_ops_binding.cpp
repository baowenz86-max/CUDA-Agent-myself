#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_embedding_launcher(
    const int64_t* indices_cpu,
    const float* weight_cpu,
    float* output_gpu,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
);

extern "C" void custom_baddbmm_launcher(
    const float* x_gpu,
    const float* batch1_cpu,
    const float* batch2_cpu,
    float* output_gpu,
    int batch,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

// ====================== custom_embedding ======================
torch::Tensor custom_embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Check indices
    TORCH_CHECK(indices.is_cpu(), "indices must be a CPU tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    // Check weight
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // Extract dimensions
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch_size, seq_len]");
    int64_t batch_size = indices.size(0);
    int64_t seq_len = indices.size(1);
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    // Allocate output on CUDA
    auto output = torch::empty({batch_size, seq_len, embedding_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_embedding_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(seq_len),
        static_cast<int>(embedding_dim),
        static_cast<int>(num_embeddings),
        stream
    );

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_embedding_launcher failed");

    return output;
}

// ====================== custom_baddbmm ======================
torch::Tensor custom_baddbmm_forward(torch::Tensor x, torch::Tensor batch1, torch::Tensor batch2) {
    // Check x
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    // Check batch1
    TORCH_CHECK(batch1.is_cpu(), "batch1 must be a CPU tensor");
    TORCH_CHECK(batch1.dtype() == torch::kFloat32, "batch1 must be float32");
    TORCH_CHECK(batch1.is_contiguous(), "batch1 must be contiguous");

    // Check batch2
    TORCH_CHECK(batch2.is_cpu(), "batch2 must be a CPU tensor");
    TORCH_CHECK(batch2.dtype() == torch::kFloat32, "batch2 must be float32");
    TORCH_CHECK(batch2.is_contiguous(), "batch2 must be contiguous");

    // Extract dimensions
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, M, N]");
    int64_t batch = x.size(0);
    int64_t M = x.size(1);
    int64_t N = x.size(2);

    TORCH_CHECK(batch1.dim() == 3 && batch1.size(0) == batch && batch1.size(1) == M,
                "batch1 shape mismatch");
    TORCH_CHECK(batch2.dim() == 3 && batch2.size(0) == batch && batch2.size(2) == N,
                "batch2 shape mismatch");
    int64_t K = batch1.size(2);
    TORCH_CHECK(batch2.size(1) == K, "batch2 K dimension mismatch");

    // Allocate output on CUDA (same shape as x)
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_baddbmm_launcher(
        x.data_ptr<float>(),
        batch1.data_ptr<float>(),
        batch2.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        stream
    );

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_baddbmm_launcher failed");

    return output;
}

// ====================== Registration ======================
void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding", &custom_embedding_forward,
          "Custom embedding lookup (indices: int64 CPU, weight: float32 CPU) -> float32 CUDA",
          py::arg("indices"), py::arg("weight"));
    m.def("custom_baddbmm", &custom_baddbmm_forward,
          "Custom batched matrix multiply-add (x: CUDA, batch1/batch2: CPU) -> float32 CUDA",
          py::arg("x"), py::arg("batch1"), py::arg("batch2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
