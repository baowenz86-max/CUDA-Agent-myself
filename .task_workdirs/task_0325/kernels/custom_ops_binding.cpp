#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    check_cuda_error();
    return output;
}

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    check_cuda_error();
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("softplus_forward", &softplus_forward, "Softplus forward (CUDA)",
          py::arg("x"));
    m.def("relu_forward", &relu_forward, "ReLU forward (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
