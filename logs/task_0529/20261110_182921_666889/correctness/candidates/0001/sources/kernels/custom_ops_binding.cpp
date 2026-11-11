// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const float* weight,
    const int64_t* indices,
    float* output,
    int num_embeddings,
    int embedding_dim,
    int batch,
    int seq_len,
    cudaStream_t stream);

extern "C" void amax_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int batch,
    int seq_len,
    int embedding_dim,
    cudaStream_t stream);

static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor embedding_forward(torch::Tensor weight, torch::Tensor indices) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch, seq_len]");

    // 处理 strides：weight 可能非连续，做 contiguous
    auto weight_contig = weight.contiguous();
    auto indices_contig = indices.contiguous();

    int num_embeddings = weight_contig.size(0);
    int embedding_dim = weight_contig.size(1);
    int batch = indices_contig.size(0);
    int seq_len = indices_contig.size(1);

    auto output = torch::empty({batch, seq_len, embedding_dim}, weight_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    embedding_forward_launcher(
        weight_contig.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_embeddings,
        embedding_dim,
        batch,
        seq_len,
        stream
    );
    check_cuda_error("embedding_forward_launcher failed");

    return output;
}

torch::Tensor amax_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.dim() == 3, "input must be 3D [batch, seq_len, embedding_dim]");

    auto x_contig = x.contiguous();
    int batch = x_contig.size(0);
    int seq_len = x_contig.size(1);
    int embedding_dim = x_contig.size(2);

    auto output = torch::empty({batch, 1, embedding_dim}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    amax_dim1_keepdim_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        seq_len,
        embedding_dim,
        stream
    );
    check_cuda_error("amax_dim1_keepdim_launcher failed");

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward (CUDA)",
          py::arg("weight"), py::arg("indices"));
    m.def("amax_dim1_keepdim", &amax_dim1_keepdim, "Amax over dim=1 with keepdim (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
