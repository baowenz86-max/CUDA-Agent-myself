// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void bitwise_xor_fp32_custom_launcher(
    const float* x,
    const float* y,
    float* out,
    int64_t numel,
    cudaStream_t stream);

torch::Tensor bitwise_xor_fp32_custom(torch::Tensor x, torch::Tensor y) {
    // Parameter checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    // Ensure contiguous (kernel assumes row-major contiguous)
    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();

    int64_t numel = x_contig.numel();
    auto out = torch::empty_like(x_contig);

    if (numel == 0) {
        return out;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bitwise_xor_fp32_custom_launcher(
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_bitwise_xor_fp32_custom(pybind11::module& m) {
    m.def("bitwise_xor_fp32_custom", &bitwise_xor_fp32_custom,
          "Bitwise XOR of abs(x) and y after int64 conversion (float32 output)",
          pybind11::arg("x"), pybind11::arg("y"));
}

REGISTER_BINDING(bitwise_xor_fp32_custom, register_bitwise_xor_fp32_custom);
