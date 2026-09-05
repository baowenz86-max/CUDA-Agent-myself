#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "../binding_registry.h"

extern "C" void fused_activation_launcher(float* output, const float* input,
                                            size_t n, float negative_slope,
                                            float exponent, float abs_max,
                                            cudaStream_t stream);

torch::Tensor fused_activation(torch::Tensor input, double negative_slope,
                               double exponent, double abs_max) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float,
                "input must be float32");

    auto output = torch::empty_like(input);
    fused_activation_launcher(
        output.data_ptr<float>(), input.data_ptr<float>(), input.numel(),
        static_cast<float>(negative_slope), static_cast<float>(exponent),
        static_cast<float>(abs_max), c10::cuda::getCurrentCUDAStream().stream());
    return output;
}

void register_fused_activation(pybind11::module& m) {
    m.def("fused_activation", &fused_activation, "Fused activation chain");
}

REGISTER_BINDING(fused_activation, register_fused_activation);
