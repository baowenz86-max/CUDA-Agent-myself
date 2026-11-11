#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void hardswish_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

// PyTorch wrapper for hardswish forward
torch::Tensor hardswish_forward(torch::Tensor x) {
    // Validate tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // x can be non-contiguous; launcher uses linear indexing over numel.
    // Output must be allocated with same shape and device.
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    hardswish_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Register function
void register_hardswish_forward(pybind11::module& m) {
    m.def("hardswish_forward", &hardswish_forward,
          "Apply Hardswish activation elementwise.\n\n"
          "Args:\n"
          "    x (torch.Tensor): Input tensor of shape [16, 512, 64000], dtype torch.float32 on CUDA.\n\n"
          "Returns:\n"
          "    torch.Tensor: Output tensor with same shape and dtype.",
          py::arg("x"));
}

REGISTER_BINDING(hardswish_forward, register_hardswish_forward);
