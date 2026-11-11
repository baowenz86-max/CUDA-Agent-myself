#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void relu_forward_launcher(float* out, const float* in, int64_t numel, cudaStream_t stream);

// PyTorch wrapper for ReLU forward
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward", pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);