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
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t weight_stride_row,
    int64_t weight_stride_col,
    cudaStream_t stream
);

extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t input_stride_embed,
    int64_t output_stride_batch,
    int64_t output_stride_seq,
    int64_t output_stride_embed,
    cudaStream_t stream
);

torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight, int64_t padding_idx) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, seq_len]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");

    int64_t batch = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");
    TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");

    auto output = torch::empty({batch, seq_len, embedding_dim}, input.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        seq_len,
        embedding_dim,
        num_embeddings,
        padding_idx,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        stream
    );

    return output;
}

torch::Tensor log_softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, seq_len, embedding_dim]");
    TORCH_CHECK(dim == 1, "only dim=1 is supported for this task");

    int64_t batch = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t embedding_dim = input.size(2);

    auto output = torch::empty_like(input, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        seq_len,
        embedding_dim,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding lookup with padding_idx",
          py::arg("input"), py::arg("weight"), py::arg("padding_idx"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "Log softmax along dimension 1",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
