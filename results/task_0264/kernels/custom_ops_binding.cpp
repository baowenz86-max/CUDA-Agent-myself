#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_forward_launcher(
    const float* x,
    float* y,
    int B,
    int D,
    int S,
    long long sx_b,
    long long sx_d,
    long long sx_s,
    long long sy_b,
    long long sy_d,
    long long sy_s,
    cudaStream_t stream);

extern "C" void argmax_forward_launcher(
    const float* x,
    int64_t* out,
    int B,
    int D,
    int S,
    long long sx_b,
    long long sx_d,
    long long sx_s,
    long long so_b,
    long long so_s,
    cudaStream_t stream);

torch::Tensor log_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, D, S)");

    int64_t B = x.size(0);
    int64_t D = x.size(1);
    int64_t S = x.size(2);
    TORCH_CHECK(B > 0 && D > 0 && S > 0, "Dimensions must be positive");
    TORCH_CHECK(B <= INT_MAX && D <= INT_MAX && S <= INT_MAX, "Dimensions exceed int range");

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute strides in elements (PyTorch strides are in elements)
    long long sx_b = static_cast<long long>(x.stride(0));
    long long sx_d = static_cast<long long>(x.stride(1));
    long long sx_s = static_cast<long long>(x.stride(2));
    long long sy_b = static_cast<long long>(y.stride(0));
    long long sy_d = static_cast<long long>(y.stride(1));
    long long sy_s = static_cast<long long>(y.stride(2));

    log_softmax_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(D),
        static_cast<int>(S),
        sx_b, sx_d, sx_s,
        sy_b, sy_d, sy_s,
        stream
    );

    return y;
}

torch::Tensor argmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, D, S)");

    int64_t B = x.size(0);
    int64_t D = x.size(1);
    int64_t S = x.size(2);
    TORCH_CHECK(B > 0 && D > 0 && S > 0, "Dimensions must be positive");
    TORCH_CHECK(B <= INT_MAX && D <= INT_MAX && S <= INT_MAX, "Dimensions exceed int range");

    auto out = torch::empty({B, S}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute strides in elements
    long long sx_b = static_cast<long long>(x.stride(0));
    long long sx_d = static_cast<long long>(x.stride(1));
    long long sx_s = static_cast<long long>(x.stride(2));
    long long so_b = static_cast<long long>(out.stride(0));
    long long so_s = static_cast<long long>(out.stride(1));

    argmax_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        static_cast<int>(B),
        static_cast<int>(D),
        static_cast<int>(S),
        sx_b, sx_d, sx_s,
        so_b, so_s,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax along dim=1",
          pybind11::arg("x"));
    m.def("argmax_forward", &argmax_forward, "Argmax along dim=1",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
