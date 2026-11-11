#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void embedding_log_softmax_launcher(
    const int64_t* indices,
    const float* weight,
    float* embedding_out,
    float* log_softmax_out,
    int B, int S, int D, int V,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor embedding_log_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t dim) {
    // Input validation
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    // Move to CUDA if necessary (contract says CPU, but be safe)
    auto indices_cuda = input.to(torch::kCUDA);
    auto weight_cuda = weight.to(torch::kCUDA);

    // Ensure contiguous
    indices_cuda = indices_cuda.contiguous();
    weight_cuda = weight_cuda.contiguous();

    int B = indices_cuda.size(0);
    int S = indices_cuda.size(1);
    int V = weight_cuda.size(0);
    int D = weight_cuda.size(1);

    TORCH_CHECK(V > 0 && D > 0, "weight must have non-zero rows and cols");

    // Allocate output tensors on CUDA
    auto embedding_out = torch::empty({B, S, D}, indices_cuda.options().dtype(torch::kFloat32));
    auto log_softmax_out = torch::empty({B, S, D}, indices_cuda.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    embedding_log_softmax_launcher(
        indices_cuda.data_ptr<int64_t>(),
        weight_cuda.data_ptr<float>(),
        embedding_out.data_ptr<float>(),
        log_softmax_out.data_ptr<float>(),
        B, S, D, V,
        stream);

    return embedding_out, log_softmax_out;
}

// Registration function
void register_embedding_log_softmax_forward(pybind11::module& m) {
    m.def("embedding_log_softmax_forward", &embedding_log_softmax_forward,
          "Embedding lookup followed by log_softmax along dim=1",
          py::arg("input"), py::arg("weight"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(embedding_log_softmax_forward, register_embedding_log_softmax_forward);
