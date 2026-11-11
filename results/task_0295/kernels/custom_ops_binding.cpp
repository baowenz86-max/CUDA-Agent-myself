#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_hardshrink_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void log_softmax_only_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void hardshrink_only_launcher(
    const float* input, float* output,
    long long numel,
    cudaStream_t stream);

static torch::Tensor check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    return x;
}

torch::Tensor log_softmax_hardshrink(torch::Tensor x) {
    check_input(x);
    auto output = torch::empty_like(x);
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_hardshrink_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);
    return output;
}

torch::Tensor log_softmax_only(torch::Tensor x) {
    check_input(x);
    auto output = torch::empty_like(x);
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_only_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);
    return output;
}

torch::Tensor hardshrink_only(torch::Tensor x) {
    check_input(x);
    auto output = torch::empty_like(x);
    long long numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardshrink_only_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_hardshrink", &log_softmax_hardshrink,
          "log_softmax along dim=1 followed by hardshrink(lambd=0.5)",
          pybind11::arg("x"));
    m.def("log_softmax_only", &log_softmax_only,
          "log_softmax along dim=1",
          pybind11::arg("x"));
    m.def("hardshrink_only", &hardshrink_only,
          "hardshrink(lambd=0.5)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
