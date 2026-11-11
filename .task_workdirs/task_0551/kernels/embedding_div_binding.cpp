#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_div_launcher(
    const int64_t* indices_ptr,
    const float* weight_ptr,
    float* output_ptr,
    int total_indices,
    int embedding_dim,
    cudaStream_t stream);

torch::Tensor embedding_div_forward(torch::Tensor indices, torch::Tensor weight) {
    // Validate indices: must be int64 and contiguous (after possible contiguous call)
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    // Allow both CPU and CUDA indices; if not contiguous, make contiguous
    auto indices_contig = indices.contiguous();

    // Validate weight
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");

    const int64_t total_indices = indices_contig.numel();
    const int64_t embedding_dim = weight.size(1);
    TORCH_CHECK(total_indices > 0, "indices cannot be empty");
    TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");

    // Allocate output on CUDA; shape is indices.shape + [embedding_dim]
    std::vector<int64_t> output_shape(indices_contig.sizes().begin(), indices_contig.sizes().end());
    output_shape.push_back(embedding_dim);
    auto output = torch::empty(output_shape, weight.options());

    // Move indices to CUDA if needed
    torch::Tensor indices_gpu;
    if (indices_contig.is_cuda()) {
        indices_gpu = indices_contig;
    } else {
        indices_gpu = indices_contig.to(torch::Device(torch::kCUDA, weight.get_device()), torch::kInt64);
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_div_launcher(
        indices_gpu.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total_indices),
        static_cast<int>(embedding_dim),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_div_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_embedding_div(pybind11::module& m) {
    m.def("embedding_div_forward", &embedding_div_forward,
          "Embedding lookup and division by 2.0",
          py::arg("indices"), py::arg("weight"));
}

REGISTER_BINDING(embedding_div, register_embedding_div);