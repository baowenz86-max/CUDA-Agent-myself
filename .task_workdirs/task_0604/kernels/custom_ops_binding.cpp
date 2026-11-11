// Binding for abs_relu6_forward custom CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void abs_relu6_forward_launcher(const float* input,
                                          float* output,
                                          long long numel,
                                          cudaStream_t stream);

// PyTorch wrapper function
torch::Tensor abs_relu6_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    // Allocate output tensor with same shape and options
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call the launcher
    abs_relu6_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// Registration function
void register_abs_relu6_forward(pybind11::module& m) {
    m.def("abs_relu6_forward", &abs_relu6_forward,
          "Applies abs then ReLU6 elementwise",
          pybind11::arg("input"));
}

// Auto-register
REGISTER_BINDING(abs_relu6_forward, register_abs_relu6_forward);
