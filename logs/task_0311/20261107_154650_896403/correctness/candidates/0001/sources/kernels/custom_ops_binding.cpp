#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Launcher declaration matches custom_ops.cu
extern "C" void leaky_relu_forward_launcher(const float* input,
                                            float* output,
                                            int64_t numel,
                                            float negative_slope,
                                            cudaStream_t stream);

// PyTorch binding for leaky_relu_forward
torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must have dtype torch.float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Allocate output tensor with same shape and options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointers
    leaky_relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        static_cast<float>(negative_slope),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward (CUDA)",
          py::arg("x"),
          py::arg("negative_slope"));
}

// Auto-register binding
REGISTER_BINDING(custom_ops, register_custom_ops);
