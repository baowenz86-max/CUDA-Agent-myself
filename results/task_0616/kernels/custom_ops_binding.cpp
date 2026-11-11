#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void sigmoid_elementwise_launcher(
    const float* x,
    float* out,
    int total,
    cudaStream_t stream);

// Wrapper for sum_dim1_keepdim
torch::Tensor sum_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sum_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim1_keepdim kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for sigmoid_elementwise
torch::Tensor sigmoid_elementwise(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, 1, H, W)");
    TORCH_CHECK(x.size(1) == 1, "Second dimension must be 1 (keepdim output)");

    int64_t total = x.numel();
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_elementwise_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_elementwise kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1_keepdim", &sum_dim1_keepdim, "Sum over channel dim (dim=1) with keepdim", py::arg("x"));
    m.def("sigmoid_elementwise", &sigmoid_elementwise, "Elementwise sigmoid", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
