#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(const int64_t* indices,
                                           const float* weight,
                                           float* output,
                                           int total_indices,
                                           int embedding_dim,
                                           cudaStream_t stream);

extern "C" void layer_norm_forward_launcher(const float* input,
                                            float* output,
                                            int total_rows,
                                            int dim,
                                            float eps,
                                            cudaStream_t stream);

// Helper to check CUDA errors after launch
static void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error: ", cudaGetErrorString(err));
}

torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int64_t total_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);

    TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");

    auto output = torch::empty({indices.size(0), indices.size(1), embedding_dim},
                               weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total_indices),
        static_cast<int>(embedding_dim),
        stream
    );

    check_cuda_error("embedding_forward");
    return output;
}

torch::Tensor layer_norm_forward(torch::Tensor input, int64_t normalized_shape) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(normalized_shape > 0, "normalized_shape must be positive");

    int64_t numel = input.numel();
    TORCH_CHECK(numel % normalized_shape == 0, "input numel must be divisible by normalized_shape");
    int64_t total_rows = numel / normalized_shape;

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float eps = 1e-5f; // matches reference LayerNorm default eps

    layer_norm_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total_rows),
        static_cast<int>(normalized_shape),
        eps,
        stream
    );

    check_cuda_error("layer_norm_forward");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding forward (gather rows)",
          py::arg("indices"), py::arg("weight"));

    m.def("layer_norm_forward", &layer_norm_forward,
          "Layer normalization without affine",
          py::arg("input"), py::arg("normalized_shape"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
