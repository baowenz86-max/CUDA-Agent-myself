#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void tanh_scale_launcher(
    void* output,
    const void* input,
    float scaling_factor,
    int n,
    int dtype_size,
    cudaStream_t stream
);

torch::Tensor tanh_scale_forward(
    torch::Tensor input,
    double scaling_factor)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const auto dtype = input.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat64 ||
        dtype == torch::kFloat16 || dtype == torch::kBFloat16,
        "Unsupported dtype: ", dtype
    );
    
    auto output = torch::empty_like(input);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    int dtype_size = static_cast<int>(input.element_size());
    int n = static_cast<int>(input.numel());
    
    tanh_scale_launcher(
        output.data_ptr(),
        input.data_ptr(),
        static_cast<float>(scaling_factor),
        n,
        dtype_size,
        stream
    );
    
    return output;
}

void register_tanh_scale(pybind11::module& m) {
    m.def("tanh_scale_forward", &tanh_scale_forward,
          "Apply tanh(x * scaling_factor) element-wise",
          py::arg("input"),
          py::arg("scaling_factor"));
}

REGISTER_BINDING(tanh_scale, register_tanh_scale);
