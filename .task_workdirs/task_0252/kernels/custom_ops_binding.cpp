#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" cudaError_t softmax_dim2_launcher(
    float* output,
    const float* input,
    int B,
    int H,
    int W,
    cudaStream_t stream
);

extern "C" cudaError_t elu_launcher(
    float* output,
    const float* input,
    int numel,
    float alpha,
    cudaStream_t stream
);

namespace {

void check_cuda_float_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor softmax_dim2(torch::Tensor x) {
    check_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, H, W)");

    int B = static_cast<int>(x.size(0));
    int H = static_cast<int>(x.size(1));
    int W = static_cast<int>(x.size(2));

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = softmax_dim2_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        B,
        H,
        W,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "softmax_dim2_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor elu(torch::Tensor x, double alpha = 1.0) {
    check_cuda_float_contiguous(x, "x");
    TORCH_CHECK(alpha >= 0.0, "alpha must be non-negative");

    int numel = static_cast<int>(x.numel());
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = elu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        static_cast<float>(alpha),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "elu_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("softmax_dim2", &softmax_dim2,
          "Softmax over last dimension (dim=2) for 3D tensor",
          py::arg("x"));
    m.def("elu", &elu,
          "Elementwise ELU",
          py::arg("x"),
          py::arg("alpha") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
