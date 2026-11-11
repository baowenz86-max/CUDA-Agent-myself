#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers exported from custom_ops.cu
extern "C" void custom_embedding_layernorm_launcher(
    const int64_t* indices,
    const float* embedding_weight,
    const float* gamma,
    const float* beta,
    float eps,
    int num_embeddings,
    int embedding_dim,
    int total_tokens,
    float* output,
    cudaStream_t stream
);

extern "C" void custom_bucketize_launcher(
    const float* input_values,
    const float* boundaries,
    int num_boundaries,
    int total_elements,
    int64_t* output,
    cudaStream_t stream
);

namespace {

torch::Tensor custom_embedding_layernorm_forward(
    torch::Tensor indices,
    torch::Tensor embedding_weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps,
    int64_t num_embeddings,
    int64_t embedding_dim
) {
    // Validate indices
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [B, L]");

    // Validate embedding weight
    TORCH_CHECK(embedding_weight.is_cuda(), "embedding_weight must be a CUDA tensor");
    TORCH_CHECK(embedding_weight.dtype() == torch::kFloat32, "embedding_weight must be float32");
    TORCH_CHECK(embedding_weight.dim() == 2, "embedding_weight must be 2D");
    TORCH_CHECK(embedding_weight.size(0) == num_embeddings, "embedding_weight row count mismatch");
    TORCH_CHECK(embedding_weight.size(1) == embedding_dim, "embedding_weight column count mismatch");

    // Validate gamma and beta
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(gamma.size(0) == embedding_dim, "gamma size mismatch");

    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");
    TORCH_CHECK(beta.size(0) == embedding_dim, "beta size mismatch");

    // Kernel requires contiguous int64 indices
    auto indices_contig = indices.contiguous();
    int64_t B = indices.size(0);
    int64_t L = indices.size(1);
    int64_t total_tokens = B * L;

    // Check integer limits for kernel int parameters
    TORCH_CHECK(num_embeddings > 0 && num_embeddings <= INT32_MAX, "num_embeddings out of range");
    TORCH_CHECK(embedding_dim > 0 && embedding_dim <= INT32_MAX, "embedding_dim out of range");
    TORCH_CHECK(total_tokens > 0 && total_tokens <= INT32_MAX, "total_tokens out of range");

    // weight, gamma, beta must be contiguous for direct pointer use
    auto weight_contig = embedding_weight.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto beta_contig = beta.contiguous();

    int num_embeddings_int = static_cast<int>(num_embeddings);
    int embedding_dim_int = static_cast<int>(embedding_dim);
    int total_tokens_int = static_cast<int>(total_tokens);

    auto output = torch::empty({B, L, embedding_dim}, indices.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_embedding_layernorm_launcher(
        indices_contig.data_ptr<int64_t>(),
        weight_contig.data_ptr<float>(),
        gamma_contig.data_ptr<float>(),
        beta_contig.data_ptr<float>(),
        static_cast<float>(eps),
        num_embeddings_int,
        embedding_dim_int,
        total_tokens_int,
        output.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_embedding_layernorm kernel launch failed");

    return output;
}

torch::Tensor custom_bucketize_forward(
    torch::Tensor input_values,
    torch::Tensor boundaries
) {
    // Validate input_values
    TORCH_CHECK(input_values.is_cuda(), "input_values must be a CUDA tensor");
    TORCH_CHECK(input_values.dtype() == torch::kFloat32, "input_values must be float32");
    TORCH_CHECK(input_values.dim() == 2, "input_values must be 2D [B, L]");

    // Validate boundaries
    TORCH_CHECK(boundaries.is_cuda(), "boundaries must be a CUDA tensor");
    TORCH_CHECK(boundaries.dtype() == torch::kFloat32, "boundaries must be float32");
    TORCH_CHECK(boundaries.dim() == 1, "boundaries must be 1D");
    TORCH_CHECK(boundaries.size(0) > 0, "boundaries must not be empty");

    auto input_contig = input_values.contiguous();
    auto boundaries_contig = boundaries.contiguous();

    int64_t B = input_values.size(0);
    int64_t L = input_values.size(1);
    int64_t total_elements = B * L;
    int64_t num_boundaries = boundaries.size(0);

    TORCH_CHECK(total_elements > 0 && total_elements <= INT32_MAX, "total_elements out of range");
    TORCH_CHECK(num_boundaries > 0 && num_boundaries <= INT32_MAX, "num_boundaries out of range");

    int total_elements_int = static_cast<int>(total_elements);
    int num_boundaries_int = static_cast<int>(num_boundaries);

    auto output = torch::empty({B, L}, input_values.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_bucketize_launcher(
        input_contig.data_ptr<float>(),
        boundaries_contig.data_ptr<float>(),
        num_boundaries_int,
        total_elements_int,
        output.data_ptr<int64_t>(),
        stream
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_bucketize kernel launch failed");

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def(
        "custom_embedding_layernorm",
        &custom_embedding_layernorm_forward,
        "Embedding lookup followed by LayerNorm",
        py::arg("indices"),
        py::arg("embedding_weight"),
        py::arg("gamma"),
        py::arg("beta"),
        py::arg("eps"),
        py::arg("num_embeddings"),
        py::arg("embedding_dim")
    );
    m.def(
        "custom_bucketize",
        &custom_bucketize_forward,
        "torch.bucketize equivalent with default right=False",
        py::arg("input_values"),
        py::arg("boundaries")
    );
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
