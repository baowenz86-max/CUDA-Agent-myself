#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void softplus_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

// PyTorch binding wrapper
torch::Tensor softplus_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "softplus_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softplus_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "softplus_forward: input must be contiguous");

    // Allocate output with same shape, dtype, device, and stride (contiguous input -> contiguous output)
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel via launcher
    softplus_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream);

    return output;
}

// Registration
void register_softplus(pybind11::module& m) {
    m.def("softplus_forward", &softplus_forward,
          "Softplus forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(softplus, register_softplus);
