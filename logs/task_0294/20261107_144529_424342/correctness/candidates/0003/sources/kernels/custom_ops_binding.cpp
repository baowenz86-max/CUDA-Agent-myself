#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void embedding_lookup_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t total_indices,
    int embedding_dim,
    int weight_row_stride,
    int weight_col_stride,
    int padding_idx,
    cudaStream_t stream);

// PyTorch wrapper for embedding lookup forward
torch::Tensor embedding_lookup_forward(
    torch::Tensor indices,
    torch::Tensor weight,
    int padding_idx = 0) {
    // Input validation
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dim() >= 1, "indices must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(indices.device() == weight.device(), "indices and weight must be on same device");

    const int64_t total_indices = indices.numel();
    const int embedding_dim = weight.size(1);

    TORCH_CHECK(weight.stride(0) > 0 && weight.stride(1) > 0, "weight strides must be positive");
    TORCH_CHECK(weight.stride(0) < INT_MAX && weight.stride(1) < INT_MAX, "weight strides too large");
    int weight_row_stride = static_cast<int>(weight.stride(0));
    int weight_col_stride = static_cast<int>(weight.stride(1));

    std::vector<int64_t> output_sizes = indices.sizes().vec();
    output_sizes.push_back(embedding_dim);
    auto output = torch::empty(output_sizes, indices.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_lookup_forward_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        total_indices,
        embedding_dim,
        weight_row_stride,
        weight_col_stride,
        padding_idx,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_embedding_lookup_forward(pybind11::module& m) {
    m.def("embedding_lookup_forward", &embedding_lookup_forward,
          "Embedding lookup forward (CUDA)",
          pybind11::arg("indices"),
          pybind11::arg("weight"),
          pybind11::arg("padding_idx") = 0);
}

// Manual registration
static BindingRegistrar _registrar_embedding_lookup_forward(
    "embedding_lookup_forward",
    [](pybind11::module& m) { register_embedding_lookup_forward(m); });
