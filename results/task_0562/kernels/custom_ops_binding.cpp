#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void round_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

// Wrapper for abs_forward
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure contiguous for element-wise kernel
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_forward CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Wrapper for round_forward
torch::Tensor round_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    round_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "round_forward CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("round_forward", &round_forward, "Element-wise round to nearest integer",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
