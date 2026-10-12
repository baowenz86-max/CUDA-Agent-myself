#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void repeat_interleave_launcher(
    void* output,
    const void* input,
    int B, int S, int E, int repeats,
    int dtype_size,
    cudaStream_t stream
);

torch::Tensor repeat_interleave_forward(
    torch::Tensor input,
    int64_t repeats,
    int64_t dim) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(repeats > 0, "repeats must be positive");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, S, E)");
    
    const auto dtype = input.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
        dtype == torch::kBFloat16 || dtype == torch::kFloat64,
        "Unsupported dtype: ", dtype
    );
    
    auto B = input.size(0);
    auto S = input.size(1);
    auto E = input.size(2);
    
    // Allocate output
    auto output = torch::empty(
        {B, S * repeats, E},
        input.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Determine dtype size in bytes
    int dtype_size = static_cast<int>(input.element_size());
    
    // Call launcher
    repeat_interleave_launcher(
        output.data_ptr(),
        input.data_ptr(),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(E),
        static_cast<int>(repeats),
        dtype_size,
        stream
    );
    
    return output;
}

void register_repeat_interleave(pybind11::module& m) {
    m.def("repeat_interleave_forward", &repeat_interleave_forward,
          "Repeat interleave along dim=1",
          py::arg("input"),
          py::arg("repeats"),
          py::arg("dim"));
}

REGISTER_BINDING(repeat_interleave, register_repeat_interleave);
