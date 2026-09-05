#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void fused_bn_digamma_max_launcher(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, float*, int64_t, int64_t, int, cudaStream_t);

torch::Tensor fused_bn_digamma_max(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor parameter) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.dim() == 5,
                "input must be a 5D float32 tensor");
    TORCH_CHECK(input.size(1) == 10, "input must have 10 channels");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda() && running_mean.is_cuda() &&
                running_var.is_cuda() && parameter.is_cuda(),
                "all parameters must be CUDA tensors");

    // Enough CTAs to saturate large GPUs while leaving ample work per CTA.
    constexpr int partial_count = 512;
    auto partial = torch::empty({partial_count}, input.options());
    auto output = torch::empty({10}, input.options());
    const int64_t channel_stride = input.size(2) * input.size(3) * input.size(4);
    fused_bn_digamma_max_launcher(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        parameter.data_ptr<float>(), partial.data_ptr<float>(),
        output.data_ptr<float>(), input.numel(), channel_stride, partial_count,
        c10::cuda::getCurrentCUDAStream().stream());
    return output;
}

void register_fused_bn_digamma_max(pybind11::module& m) {
    m.def("fused_bn_digamma_max", &fused_bn_digamma_max);
}
REGISTER_BINDING(fused_bn_digamma_max, register_fused_bn_digamma_max);
