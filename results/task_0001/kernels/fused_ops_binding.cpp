#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void fused_ops_launcher(float* output, const float* input, int64_t numel, float negative_slope, float pow_exponent, float mish_alpha, float abs_max, cudaStream_t stream);

torch::Tensor fused_ops_forward(torch::Tensor input, double negative_slope, double pow_exponent, double mish_alpha, double relu_threshold, double abs_max) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_ops_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        static_cast<float>(negative_slope),
        static_cast<float>(pow_exponent),
        static_cast<float>(mish_alpha),
        static_cast<float>(abs_max),
        stream
    );

    return output;
}

void register_fused_ops(pybind11::module& m) {
    m.def("fused_ops_forward", &fused_ops_forward,
          "Fused LeakyReLU+Pow+Abs+Mish+ReLU+Clamp",
          py::arg("input"),
          py::arg("negative_slope"),
          py::arg("pow_exponent"),
          py::arg("mish_alpha"),
          py::arg("relu_threshold"),
          py::arg("abs_max"));
}

REGISTER_BINDING(fused_ops, register_fused_ops);
