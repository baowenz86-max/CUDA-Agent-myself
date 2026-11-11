#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_dim1_forward_launcher(const float* input, float* output, int B, int H, int W, cudaStream_t stream);
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

// Softmax over dimension 1 wrapper
torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, W,
        stream
    );

    return output;
}

// ReLU wrapper
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax over dimension 1",
          py::arg("x"));
    m.def("relu_forward", &relu_forward, "ReLU activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
