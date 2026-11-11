#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_isnan_launcher(float* output,
                                      const float* input,
                                      long long numel,
                                      cudaStream_t stream);

extern "C" void custom_relu6_launcher(float* output,
                                       const float* input,
                                       long long numel,
                                       cudaStream_t stream);

// PyTorch wrapper for isnan
torch::Tensor custom_isnan_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_isnan: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_isnan: input must be float32");

    auto output = torch::empty_like(x);
    long long numel = static_cast<long long>(x.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_isnan_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// PyTorch wrapper for relu6
torch::Tensor custom_relu6_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_relu6: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_relu6: input must be float32");

    auto output = torch::empty_like(x);
    long long numel = static_cast<long long>(x.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_relu6_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_isnan", &custom_isnan_forward, "Elementwise isnan",
          py::arg("x"));
    m.def("custom_relu6", &custom_relu6_forward, "Elementwise ReLU6",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);