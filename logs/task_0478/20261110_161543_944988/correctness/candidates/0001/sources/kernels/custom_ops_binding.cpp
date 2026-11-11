#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_softmax_lastdim_v2_launcher(
    float* output,
    const float* input,
    int N,
    int H,
    int W,
    cudaStream_t stream);

extern "C" void custom_mish_launcher(
    float* output,
    const float* input,
    long total,
    cudaStream_t stream);

// Softmax along dim=1 for contiguous [N,H,W] float32 CUDA tensor
torch::Tensor custom_softmax_lastdim_v2(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_softmax_lastdim_v2: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax_lastdim_v2: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_softmax_lastdim_v2: input must have 3 dimensions [N,H,W]");
    TORCH_CHECK(x.is_contiguous(), "custom_softmax_lastdim_v2: input must be contiguous");

    int N = x.size(0);
    int H = x.size(1);
    int W = x.size(2);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_lastdim_v2_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        H,
        W,
        stream);

    return output;
}

// Mish activation elementwise
torch::Tensor custom_mish(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mish: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mish: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_mish: input must be contiguous");

    auto output = torch::empty_like(x);
    long total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_mish_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        total,
        stream);

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_lastdim_v2", &custom_softmax_lastdim_v2, "Softmax along dim=1 for contiguous CUDA float32 tensor", py::arg("x"));
    m.def("custom_mish", &custom_mish, "Mish activation elementwise", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);