#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void embedding_lookup_launcher(
    const float* weight,
    const int64_t* input_ids,
    float* output,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t batch_size,
    int64_t seq_len,
    cudaStream_t stream
);

extern "C" void index_select_launcher(
    const float* input,
    int64_t dim,
    const int64_t* index,
    float* output,
    int64_t batch_size,
    int64_t src_len,
    int64_t index_len,
    int64_t embedding_dim,
    cudaStream_t stream
);

// embedding_lookup_forward: output[b, s, d] = weight[input_ids[b, s], d]
torch::Tensor embedding_lookup_forward(
    torch::Tensor weight,
    torch::Tensor input_ids
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input_ids.is_contiguous(), "input_ids must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input_ids.dtype() == torch::kInt64, "input_ids must be int64");

    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(input_ids.dim() == 2, "input_ids must be 2D [batch_size, seq_len]");

    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);
    int64_t batch_size = input_ids.size(0);
    int64_t seq_len = input_ids.size(1);

    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(seq_len > 0, "seq_len must be positive");
    TORCH_CHECK(embedding_dim % 4 == 0, "embedding_dim must be divisible by 4 for vectorized kernel");

    auto output = torch::empty({batch_size, seq_len, embedding_dim}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_lookup_launcher(
        weight.data_ptr<float>(),
        input_ids.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_embeddings,
        embedding_dim,
        batch_size,
        seq_len,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_lookup_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// index_select_forward: output[b, i, d] = input[b, index[i], d] for dim=1
torch::Tensor index_select_forward(
    torch::Tensor input,
    int64_t dim,
    torch::Tensor index
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(index.dtype() == torch::kInt64, "index must be int64");

    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, src_len, embedding_dim]");
    TORCH_CHECK(index.dim() == 1, "index must be 1D [index_len]");
    TORCH_CHECK(dim == 1, "only dim=1 is supported for this 3D input");

    int64_t batch_size = input.size(0);
    int64_t src_len = input.size(1);
    int64_t embedding_dim = input.size(2);
    int64_t index_len = index.size(0);

    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(src_len > 0, "src_len must be positive");
    TORCH_CHECK(index_len > 0, "index_len must be positive");
    TORCH_CHECK(embedding_dim % 4 == 0, "embedding_dim must be divisible by 4 for vectorized kernel");

    auto output = torch::empty({batch_size, index_len, embedding_dim}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    index_select_launcher(
        input.data_ptr<float>(),
        dim,
        index.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch_size,
        src_len,
        index_len,
        embedding_dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "index_select_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Register both operators
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_lookup_forward", &embedding_lookup_forward,
          "Embedding lookup forward: output[b, s, d] = weight[input_ids[b, s], d]",
          pybind11::arg("weight"),
          pybind11::arg("input_ids"));

    m.def("index_select_forward", &index_select_forward,
          "Index select forward along dim: output[b, i, d] = input[b, index[i], d]",
          pybind11::arg("input"),
          pybind11::arg("dim"),
          pybind11::arg("index"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
