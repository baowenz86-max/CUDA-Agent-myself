#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void relu_clamp_forward_launcher(const float* input,
                                            float* output,
                                            int64_t n,
                                            cudaStream_t stream);

torch::Tensor relu_clamp_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Allocate output with same shape and stride as input
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    relu_clamp_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

void register_relu_clamp(pybind11::module& m) {
    m.def("relu_clamp_forward", &relu_clamp_forward,
          "Fused ReLU + clamp(min=0, max=1) forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(relu_clamp, register_relu_clamp);
