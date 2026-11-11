#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_launcher(const int64_t* indices,
                                   const float* weight,
                                   float* output,
                                   int64_t num_indices,
                                   int64_t embedding_dim,
                                   cudaStream_t stream);

extern "C" void clamp_min_launcher(float* input,
                                   float* output,
                                   int64_t numel,
                                   float min_value,
                                   cudaStream_t stream);

// Wrapper for embedding_forward
// indices: int64 tensor of shape [N] (flattened), may be CPU
// weight: float32 tensor of shape [num_embeddings, embedding_dim], may be CPU
torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Validate dtypes
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");

    // Ensure CUDA tensors (move if CPU)
    if (!indices.is_cuda()) {
        indices = indices.cuda();
    }
    if (!weight.is_cuda()) {
        weight = weight.cuda();
    }

    // Ensure contiguous
    indices = indices.contiguous();
    weight = weight.contiguous();

    // Get dimensions
    int64_t num_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(embedding_dim > 0, "weight must have positive second dimension");

    // Allocate output: shape = indices.shape + [embedding_dim]
    std::vector<int64_t> out_sizes = indices.sizes().vec();
    out_sizes.push_back(embedding_dim);
    auto output = torch::empty(out_sizes, weight.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_launcher(indices.data_ptr<int64_t>(),
                       weight.data_ptr<float>(),
                       output.data_ptr<float>(),
                       num_indices,
                       embedding_dim,
                       stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for clamp_min_forward
// input: float32 tensor of arbitrary shape, must be CUDA
torch::Tensor clamp_min_forward(torch::Tensor input, double min_value) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    input = input.contiguous();

    int64_t numel = input.numel();
    auto output = torch::empty_like(input);

    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_min_launcher(input.data_ptr<float>(),
                       output.data_ptr<float>(),
                       numel,
                       static_cast<float>(min_value),
                       stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_min_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding lookup forward",
          py::arg("indices"), py::arg("weight"));
    m.def("clamp_min_forward", &clamp_min_forward,
          "Clamp minimum forward",
          py::arg("input"), py::arg("min_value") = 0.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
