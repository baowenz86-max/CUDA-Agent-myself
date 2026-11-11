#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void log_softmax_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

// PyTorch wrapper for element-wise absolute value
torch::Tensor abs_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// PyTorch wrapper for log_softmax over entire tensor
torch::Tensor log_softmax_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax over entire tensor",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
