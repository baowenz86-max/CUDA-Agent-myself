// PyTorch binding for sigmoid forward CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream);

// PyTorch tensor wrapper
torch::Tensor sigmoid_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output same shape
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    sigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    // Optional: check for kernel errors (async errors are not caught here,
    // but sync check can be done with cudaGetLastError if needed)
    return output;
}

// Register with pybind
void register_sigmoid_forward(pybind11::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward,
          "Elementwise sigmoid forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(sigmoid_forward, register_sigmoid_forward);
