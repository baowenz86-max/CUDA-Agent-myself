#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "../binding_registry.h"

extern "C" void fused_activation_launcher(
    float*, const float*, long long, float, float, float, cudaStream_t);

torch::Tensor fused_activation(
    torch::Tensor input,
    double negative_slope,
    double exponent,
    double upper_bound) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_activation_launcher(
        output.data_ptr<float>(), input.data_ptr<float>(), input.numel(),
        static_cast<float>(negative_slope), static_cast<float>(exponent),
        static_cast<float>(upper_bound), stream);
    return output;
}

void register_fused_activation(pybind11::module& m) {
    m.def("fused_activation", &fused_activation);
}

REGISTER_BINDING(fused_activation, register_fused_activation);
