#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream);

extern "C" void hardswish_forward_launcher(const float* input,
                                            float* output,
                                            int64_t numel,
                                            cudaStream_t stream);

namespace {

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return output;
}

torch::Tensor hardswish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardswish_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardswish_forward: input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardswish_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward",
          pybind11::arg("x"));
    m.def("hardswish_forward", &hardswish_forward, "Hardswish activation forward",
          pybind11::arg("x"));
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
