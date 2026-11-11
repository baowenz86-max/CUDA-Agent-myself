#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(
    float* out,
    const float* in,
    int64_t numel,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
);

extern "C" void custom_reduce_lastdim_launcher(
    float* out,
    const float* in,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
);

// Wrapper for custom_abs
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t numel = x.numel();
    auto out = torch::empty(x.sizes(), x.options());

    const int64_t block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        block_size,
        grid_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for custom_reduce_lastdim
torch::Tensor custom_reduce_lastdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out = torch::empty({B, C, H}, x.options());

    const int64_t block_size = 256;
    int64_t total = B * C * H;
    int64_t grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_reduce_lastdim_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        B,
        C,
        H,
        W,
        block_size,
        grid_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_reduce_lastdim kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Element-wise absolute value",
          pybind11::arg("x"));
    m.def("custom_reduce_lastdim", &custom_reduce_lastdim, "Sum over last dimension",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
