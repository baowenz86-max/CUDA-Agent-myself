#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// External launcher declarations from custom_ops.cu
extern "C" void custom_identity_launcher(
    float* output,
    const float* input,
    long long total_elements,
    cudaStream_t stream
);

extern "C" void custom_elu_launcher(
    float* output,
    const float* input,
    float alpha,
    long long total_elements,
    cudaStream_t stream
);

// Helper to check common tensor requirements
static void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.numel() > 0, name, " must have at least one element");
}

// custom_identity: returns a tensor identical to input (same strides, shape, values)
torch::Tensor custom_identity(torch::Tensor x) {
    check_tensor(x, "x");
    
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_identity_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_identity kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// custom_elu: computes ELU elementwise, output same shape/strides as input
torch::Tensor custom_elu(torch::Tensor x, double alpha) {
    check_tensor(x, "x");
    TORCH_CHECK(alpha > 0, "alpha must be positive");
    
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_elu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<float>(alpha),
        static_cast<long long>(x.numel()),
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_elu kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_identity", &custom_identity, "Identity copy preserving strides",
          py::arg("x"));
    m.def("custom_elu", &custom_elu, "Elementwise ELU",
          py::arg("x"), py::arg("alpha") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
