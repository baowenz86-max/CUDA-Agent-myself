#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void clone_forward_launcher(
    float* out,
    const float* in,
    int64_t numel,
    cudaStream_t stream
);

// PyTorch binding for clone_forward
torch::Tensor clone_forward(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Allocate output with same shape, dtype, device, and strides
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch the copy kernel
    clone_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_clone_forward(pybind11::module& m) {
    m.def(
        "clone_forward",
        &clone_forward,
        "Clone a CUDA float tensor",
        py::arg("x")
    );
}

// Auto-register
REGISTER_BINDING(clone_forward, register_clone_forward);
