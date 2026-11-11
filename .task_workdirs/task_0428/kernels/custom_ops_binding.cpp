#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void abs_div_forward_launcher(const float* input,
                                         float* output,
                                         int64_t divisor,
                                         int64_t total_elements,
                                         cudaStream_t stream);

// PyTorch tensor wrapper
torch::Tensor abs_div_forward(torch::Tensor x, int64_t divisor) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure input is contiguous; launcher assumes contiguous memory
    auto x_contig = x.contiguous();

    // Allocate output tensor with same shape and options
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    abs_div_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        x_contig.numel(),
        stream
    );

    return output;
}

// Registration function
void register_abs_div_forward(pybind11::module& m) {
    m.def("abs_div_forward", &abs_div_forward,
          "Element-wise abs(x) then divide by divisor",
          py::arg("x"),
          py::arg("divisor"));
}

// Auto-register
REGISTER_BINDING(abs_div_forward, register_abs_div_forward);