#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declarations from custom_ops.cu
extern "C" void log_softmax_dim1_launcher(const float* input, float* output,
                                          int64_t N, int64_t C, int64_t H, int64_t W,
                                          cudaStream_t stream);

extern "C" void div_scalar_launcher(const float* input, float* output,
                                     double divisor, int64_t numel, cudaStream_t stream);

// Wrapper for log_softmax along dim=1
torch::Tensor log_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto output = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// Wrapper for element-wise division by scalar
torch::Tensor div_scalar(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    div_scalar_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        numel,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_dim1", &log_softmax_dim1, "Compute log_softmax along dimension 1",
          py::arg("x"));
    m.def("div_scalar", &div_scalar, "Divide tensor by scalar",
          py::arg("x"), py::arg("divisor") = 0.5);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
