#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total_elements,
                                      cudaStream_t stream);

extern "C" void softmax_forward_launcher(const float* input,
                                         float* output,
                                         int64_t outer_size,
                                         int64_t inner_size,
                                         cudaStream_t stream);

// PyTorch wrapper for ReLU
torch::Tensor relu_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Output tensor, same shape as input
    auto output = torch::empty_like(x);

    int64_t total_elements = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    return output;
}

// PyTorch wrapper for Softmax along last dimension
torch::Tensor softmax_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Output tensor, same shape as input
    auto output = torch::empty_like(x);

    // Softmax is over last dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        outer_size *= x.size(i);
    }
    int64_t inner_size = x.size(-1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        inner_size,
        stream
    );

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
    m.def("softmax_forward", &softmax_forward, "Softmax forward along last dim",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);