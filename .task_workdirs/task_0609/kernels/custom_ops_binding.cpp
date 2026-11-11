#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations for launchers from custom_ops.cu
extern "C" void isnan_launcher(bool* out, const float* in, int64_t numel, cudaStream_t stream);
extern "C" void masked_fill_launcher(float* out, const float* in, const bool* mask, float value, int64_t numel, cudaStream_t stream);

// isnan: input float32 contiguous CUDA tensor, output bool tensor same shape
torch::Tensor isnan_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() <= INT64_MAX, "Too many elements");

    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isnan_launcher(
        out.data_ptr<bool>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "isnan kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// masked_fill: input float32, mask bool, value float; output new float32 tensor
torch::Tensor masked_fill_forward(torch::Tensor x, torch::Tensor mask, double value) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "Mask must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "Mask must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be bool");
    TORCH_CHECK(x.sizes() == mask.sizes(), "Shape mismatch between input and mask");
    TORCH_CHECK(x.numel() <= INT64_MAX, "Too many elements");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_fill_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        static_cast<float>(value),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "masked_fill kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("isnan", &isnan_forward, "Detect NaN values", py::arg("x"));
    m.def("masked_fill", &masked_fill_forward, "Fill masked values with scalar", py::arg("x"), py::arg("mask"), py::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
