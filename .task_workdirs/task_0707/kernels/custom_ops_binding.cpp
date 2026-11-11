#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t N0,
    int64_t N1,
    int64_t embedding_dim,
    int64_t num_embeddings,
    cudaStream_t stream);

extern "C" void mean_dim1_launcher(
    const float* x,
    float* output,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    cudaStream_t stream);

static torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Check devices
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    // Check dtypes
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    // Check dimensions
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    // Ensure contiguity: indices may have arbitrary strides; make contiguous
    auto indices_contig = indices.contiguous();
    // weight is expected contiguous, but ensure it too
    auto weight_contig = weight.contiguous();

    int64_t N0 = indices_contig.size(0);
    int64_t N1 = indices_contig.size(1);
    int64_t embedding_dim = weight_contig.size(1);
    int64_t num_embeddings = weight_contig.size(0);

    // Allocate output [N0, N1, embedding_dim]
    auto output = torch::empty({N0, N1, embedding_dim}, weight_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        indices_contig.data_ptr<int64_t>(),
        weight_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N0,
        N1,
        embedding_dim,
        num_embeddings,
        stream
    );

    return output;
}

static torch::Tensor mean_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    // Ensure contiguous for kernel that uses flat indexing
    auto x_contig = x.contiguous();

    int64_t N0 = x_contig.size(0);
    int64_t N1 = x_contig.size(1);
    int64_t N2 = x_contig.size(2);

    // Allocate output [N0, N2]
    auto output = torch::empty({N0, N2}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim1_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N0,
        N1,
        N2,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward",
          py::arg("indices"), py::arg("weight"));
    m.def("mean_dim1_forward", &mean_dim1_forward, "Mean over dim 1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
