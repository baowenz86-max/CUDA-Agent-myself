#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void bn_digamma_max_launcher(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, int64_t, int64_t, float, cudaStream_t);

torch::Tensor bn_digamma_max(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, torch::Tensor parameter,
    double eps) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(input.dim() == 5 && input.size(1) == 10, "expected NCHW[D], C=10");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous() &&
                mean.is_contiguous() && var.is_contiguous() &&
                parameter.is_contiguous(), "parameters must be contiguous");
    auto output = torch::empty_like(parameter);
    int64_t spatial = input.size(2) * input.size(3) * input.size(4);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bn_digamma_max_launcher(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), parameter.data_ptr<float>(),
        output.data_ptr<float>(), input.numel(), spatial, (float)eps, stream);
    return output;
}

void register_bn_digamma_max(pybind11::module& m) {
    m.def("bn_digamma_max", &bn_digamma_max);
}
REGISTER_BINDING(bn_digamma_max, register_bn_digamma_max);
