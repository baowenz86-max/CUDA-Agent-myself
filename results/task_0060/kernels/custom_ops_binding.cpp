#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void fused_relu_hardsigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream);

static torch::Tensor fused_relu_hardsigmoid(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto output = torch::empty_like(x);

    fused_relu_hardsigmoid_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream);

    return output;
}

static void register_fused_relu_hardsigmoid(pybind11::module& m) {
    m.def("fused_relu_hardsigmoid", &fused_relu_hardsigmoid,
          "Fused ReLU + HardSigmoid",
          pybind11::arg("x"));
}

REGISTER_BINDING(fused_relu_hardsigmoid, register_fused_relu_hardsigmoid);