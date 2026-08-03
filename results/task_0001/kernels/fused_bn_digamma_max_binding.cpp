#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "../binding_registry.h"

extern "C" void fused_bn_digamma_max_launcher(
    float*, const float*, const float*, const float*, const float*, const float*,
    const float*, long long, int, float, cudaStream_t);

torch::Tensor fused_bn_digamma_max(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor parameter,
    double eps) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "float32 input required");
    TORCH_CHECK(input.dim() == 5 && input.size(1) == 10, "expected NCDHW with C=10");
    auto output = torch::empty_like(parameter);
    const int spatial = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_bn_digamma_max_launcher(
        output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        parameter.data_ptr<float>(), input.numel(), spatial, static_cast<float>(eps), stream);
    return output;
}

void register_fused_bn_digamma_max(pybind11::module& m) {
    m.def("fused_bn_digamma_max", &fused_bn_digamma_max);
}

REGISTER_BINDING(fused_bn_digamma_max, register_fused_bn_digamma_max);
