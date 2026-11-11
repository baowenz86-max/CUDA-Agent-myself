#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// External launcher declarations from custom_ops.cu
extern "C" void custom_identity_launcher(
    float* output,
    const float* input,
    int ndim,
    const long long* shape,
    const long long* output_strides,
    const long long* input_strides,
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

// Copy strides to a std::vector<long long>
static std::vector<long long> get_strides(const torch::Tensor& t) {
    std::vector<long long> strides(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        strides[i] = static_cast<long long>(t.stride(i));
    }
    return strides;
}

// Copy shape to a std::vector<long long>
static std::vector<long long> get_shape(const torch::Tensor& t) {
    std::vector<long long> shape(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        shape[i] = static_cast<long long>(t.size(i));
    }
    return shape;
}

// custom_identity: returns a tensor identical to input (same strides, shape, values)
torch::Tensor custom_identity(torch::Tensor x) {
    check_tensor(x, "x");
    
    // Allocate output with same shape and strides (preserving non-contiguous layout)
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());
    
    auto shape_vec = get_shape(x);
    auto out_strides_vec = get_strides(output);
    auto in_strides_vec = get_strides(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    custom_identity_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(x.dim()),
        shape_vec.data(),
        out_strides_vec.data(),
        in_strides_vec.data(),
        static_cast<long long>(x.numel()),
        stream
    );
    
    // Check for kernel launch errors
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
