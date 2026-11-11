#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_launcher(
    const float* weight,
    const int64_t* indices,
    float* output,
    int total_elements,
    int embedding_dim,
    int seq_len,
    cudaStream_t stream);

extern "C" void argsort_launcher(
    const float* input,
    int64_t* output,
    int batch,
    int seq_len,
    int embedding_dim,
    int dim,
    cudaStream_t stream);

torch::Tensor embedding_forward(torch::Tensor weight, torch::Tensor indices) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch, seq_len]");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);
    int64_t batch = indices.size(0);
    int64_t seq_len = indices.size(1);

    // Indices must be within range
    TORCH_CHECK(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < num_embeddings,
                "indices out of range");

    auto output = torch::empty({batch, seq_len, embedding_dim}, weight.options());

    int total_elements = static_cast<int>(batch * seq_len * embedding_dim);
    TORCH_CHECK(total_elements > 0, "empty output");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_launcher(
        weight.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        total_elements,
        static_cast<int>(embedding_dim),
        static_cast<int>(seq_len),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor argsort_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, seq_len, embedding_dim]");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int64_t batch = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t embedding_dim = input.size(2);

    auto output = torch::empty({batch, seq_len, embedding_dim}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    argsort_launcher(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        static_cast<int>(batch),
        static_cast<int>(seq_len),
        static_cast<int>(embedding_dim),
        static_cast<int>(dim),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "argsort_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding forward (CUDA)",
          py::arg("weight"), py::arg("indices"));
    m.def("argsort_forward", &argsort_forward,
          "Argsort forward along dim=1 (CUDA)",
          py::arg("input"), py::arg("dim") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
