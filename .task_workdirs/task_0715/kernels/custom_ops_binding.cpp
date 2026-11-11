#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers
extern "C" void abs_forward_launcher(float* out, const float* in, long long numel, cudaStream_t stream);
extern "C" void prelu_forward_launcher(float* out, const float* in, const float* weight,
                                       long long numel, long long channels, long long inner_size,
                                       cudaStream_t stream);

namespace {

void check_cuda_float(torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    t = t.contiguous();
}

} // anonymous namespace

torch::Tensor abs_forward(torch::Tensor x) {
    check_cuda_float(x, "x");
    auto out = torch::empty_like(x);
    long long numel = static_cast<long long>(x.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(out.data_ptr<float>(), x.data_ptr<float>(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_forward kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor prelu_forward(torch::Tensor x, torch::Tensor weight) {
    check_cuda_float(x, "x");
    check_cuda_float(weight, "weight");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    long long C = x.size(1);
    long long H = x.size(2);
    long long W = x.size(3);
    long long inner_size = H * W;
    TORCH_CHECK(weight.numel() == 1 || weight.numel() == C,
                "weight must have 1 element or equal to number of channels");
    long long weight_numel = weight.numel();
    // If scalar weight, we could broadcast, but here it's per-channel
    auto out = torch::empty_like(x);
    long long numel = static_cast<long long>(x.numel());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    prelu_forward_launcher(out.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
                           numel, C, inner_size, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prelu_forward kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Elementwise absolute value", pybind11::arg("x"));
    m.def("prelu_forward", &prelu_forward, "PReLU activation", pybind11::arg("x"), pybind11::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
