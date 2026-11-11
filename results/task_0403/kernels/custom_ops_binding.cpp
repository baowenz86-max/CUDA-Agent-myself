#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void embedding_lookup_forward(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t total_tokens,
    int64_t embedding_dim,
    int64_t num_embeddings,
    cudaStream_t stream);

extern "C" void l2_normalize_forward(
    const float* x,
    float* y,
    int64_t total_tokens,
    int64_t embedding_dim,
    float epsilon,
    cudaStream_t stream);

// embedding_lookup 包装函数
torch::Tensor embedding_lookup(torch::Tensor input, torch::Tensor weight) {
    // 输入索引应为 int64
    TORCH_CHECK(input.dtype() == torch::kInt64, "embedding_lookup: input must be int64");
    TORCH_CHECK(input.dim() == 2, "embedding_lookup: input must be 2D [batch_size, seq_len]");

    // 权重应为 float32 且 CUDA 连续
    TORCH_CHECK(weight.is_cuda(), "embedding_lookup: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "embedding_lookup: weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "embedding_lookup: weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "embedding_lookup: weight must be 2D [num_embeddings, embedding_dim]");

    // 如果输入在 CPU，则将其转移到 CUDA（保持 int64）
    torch::Tensor input_cuda;
    if (input.is_cuda()) {
        input_cuda = input.contiguous();
    } else {
        input_cuda = input.to(torch::kCUDA);
    }
    TORCH_CHECK(input_cuda.is_contiguous(), "embedding_lookup: input must be contiguous after transfer");

    const int64_t batch_size = input_cuda.size(0);
    const int64_t seq_len = input_cuda.size(1);
    const int64_t total_tokens = batch_size * seq_len;
    const int64_t num_embeddings = weight.size(0);
    const int64_t embedding_dim = weight.size(1);

    // 分配输出
    auto output = torch::empty({batch_size, seq_len, embedding_dim}, weight.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    embedding_lookup_forward(
        input_cuda.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        total_tokens,
        embedding_dim,
        num_embeddings,
        stream);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_lookup kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// l2_normalize 包装函数
torch::Tensor l2_normalize(torch::Tensor x, double epsilon) {
    TORCH_CHECK(x.is_cuda(), "l2_normalize: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "l2_normalize: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "l2_normalize: x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "l2_normalize: x must be 3D [batch_size, seq_len, embedding_dim]");

    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t embedding_dim = x.size(2);
    const int64_t total_tokens = batch_size * seq_len;

    // 分配输出
    auto output = torch::empty_like(x);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    l2_normalize_forward(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_tokens,
        embedding_dim,
        static_cast<float>(epsilon),
        stream);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "l2_normalize kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_lookup", &embedding_lookup,
          "Embedding lookup forward",
          py::arg("input"), py::arg("weight"));
    m.def("l2_normalize", &l2_normalize,
          "L2 normalization forward",
          py::arg("x"), py::arg("epsilon") = 1e-09);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
