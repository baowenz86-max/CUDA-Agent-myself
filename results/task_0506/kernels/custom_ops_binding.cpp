#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t in_length,
    int64_t out_length,
    int64_t pool_size,
    cudaStream_t stream);

// ReLU forward binding
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "relu_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream);

    return output;
}

// MaxPool1d forward binding
torch::Tensor max_pool1d_forward(torch::Tensor x, int64_t pool_size) {
    TORCH_CHECK(x.is_cuda(), "max_pool1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_pool1d_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "max_pool1d_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 3, "max_pool1d_forward: input must have 3 dimensions (batch, channels, length)");
    TORCH_CHECK(pool_size > 0, "max_pool1d_forward: pool_size must be positive");

    int64_t batch_size = x.size(0);
    int64_t channels = x.size(1);
    int64_t in_length = x.size(2);
    int64_t out_length = in_length / pool_size;  // floor division, trailing elements ignored
    TORCH_CHECK(out_length > 0, "max_pool1d_forward: output length must be positive");

    auto output = torch::empty({batch_size, channels, out_length}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool1d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_length,
        out_length,
        pool_size,
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward,
          "Apply ReLU elementwise: output = max(input, 0)",
          py::arg("x"));
    m.def("max_pool1d_forward", &max_pool1d_forward,
          "Apply MaxPool1d along last dimension with non-overlapping windows",
          py::arg("x"),
          py::arg("pool_size"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
