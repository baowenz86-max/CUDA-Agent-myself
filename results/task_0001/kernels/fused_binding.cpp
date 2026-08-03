#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void fused_launcher(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, float*, int64_t, int64_t, int, float, cudaStream_t);

torch::Tensor fused_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, torch::Tensor parameter, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Float, "float32 only");
    TORCH_CHECK(x.dim() == 5 && x.size(1) == 10, "expected N,C,D,H,W with C=10");

    constexpr int blocks = 256;
    auto partial = torch::empty({blocks}, x.options());
    auto output = torch::empty({10}, x.options());
    const int64_t spatial = x.size(2) * x.size(3) * x.size(4);
    fused_launcher(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), parameter.data_ptr<float>(),
        partial.data_ptr<float>(), output.data_ptr<float>(), spatial, x.numel(),
        blocks, static_cast<float>(eps),
        c10::cuda::getCurrentCUDAStream().stream());
    return output;
}

void register_fused(pybind11::module& m) {
    m.def("fused_forward", &fused_forward, "Fused BN, digamma, max, add");
}

REGISTER_BINDING(fused, register_fused);
