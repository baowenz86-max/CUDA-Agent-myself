#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cos_launcher(const float* input, float* output, long numel, cudaStream_t stream);
extern "C" void custom_identity_launcher(const float* input, float* output, long numel, cudaStream_t stream);

// Wrapper for custom_cos
torch::Tensor custom_cos_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    // Ensure contiguous; if not, make contiguous (preserves data)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_cos_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);
    
    return output;
}

// Wrapper for custom_identity
torch::Tensor custom_identity_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    // Ensure contiguous; if not, make contiguous (preserves data)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_identity_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);
    
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_cos", &custom_cos_forward, "Custom cosine forward", py::arg("x"));
    m.def("custom_identity", &custom_identity_forward, "Custom identity forward", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
