#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void subtract_scalar_launcher(const float* input,
                                         float* output,
                                         float scalar,
                                         long long num_elements,
                                         cudaStream_t stream);

// Wrapper function for subtract_scalar
torch::Tensor subtract_scalar(torch::Tensor x, torch::Tensor scalar) {
    // Check x is CUDA float32
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    // Check scalar is CPU float32 scalar tensor
    TORCH_CHECK(scalar.is_cpu(), "scalar must be a CPU tensor");
    TORCH_CHECK(scalar.dtype() == torch::kFloat32, "scalar must be float32");
    TORCH_CHECK(scalar.numel() == 1, "scalar must have exactly one element");

    // Ensure x is contiguous; if not, make contiguous copy
    torch::Tensor x_contig = x.contiguous();

    // Allocate output with same shape and options as x_contig
    torch::Tensor output = torch::empty_like(x_contig);

    // Extract scalar value
    float scalar_value = scalar.item<float>();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    subtract_scalar_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar_value,
        x_contig.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("subtract_scalar", &subtract_scalar,
          "Subtract a scalar from a tensor",
          py::arg("x"), py::arg("scalar"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
