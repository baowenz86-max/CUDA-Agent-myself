#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void mul_tensor_launcher(float* out,
                                    const float* x,
                                    const float* y,
                                    int64_t n,
                                    cudaStream_t stream);

torch::Tensor mul_tensor(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) {
        return out;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mul_tensor_launcher(out.data_ptr<float>(),
                        x.data_ptr<float>(),
                        y.data_ptr<float>(),
                        n,
                        stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mul_tensor_launcher failed");
    return out;
}

void register_mul_tensor(pybind11::module& m) {
    m.def("mul_tensor", &mul_tensor, "Element-wise multiplication",
          pybind11::arg("x"), pybind11::arg("y"));
}

REGISTER_BINDING(mul_tensor, register_mul_tensor);
