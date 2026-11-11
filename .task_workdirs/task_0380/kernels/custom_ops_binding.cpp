#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_dim1_launcher(
    const float* x_ptr,
    float* out_ptr,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void custom_leaky_relu_launcher(
    const float* x_ptr,
    float* y_ptr,
    int64_t numel,
    double negative_slope,
    cudaStream_t stream);

// Wrapper for custom_mean_dim1
torch::Tensor custom_mean_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mean_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_mean_dim1: input must be 4-dimensional [N, C, H, W]");
    TORCH_CHECK(x.is_contiguous(), "custom_mean_dim1: input must be contiguous");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "custom_mean_dim1: input dimensions must be positive");

    // Allocate output with shape [N, 1, H, W], same options as input
    auto out = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_mean_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_mean_dim1: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for custom_leaky_relu
torch::Tensor custom_leaky_relu(torch::Tensor x, double negative_slope = 0.01) {
    TORCH_CHECK(x.is_cuda(), "custom_leaky_relu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_leaky_relu: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_leaky_relu: input must be contiguous");

    int64_t numel = x.numel();
    TORCH_CHECK(numel > 0, "custom_leaky_relu: input must have at least one element");

    // Allocate output with same shape and options
    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_leaky_relu_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        negative_slope,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_leaky_relu: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Register both functions with pybind11
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim1", &custom_mean_dim1,
          "Mean over dimension 1 with keepdim=True",
          py::arg("x"));
    m.def("custom_leaky_relu", &custom_leaky_relu,
          "Leaky ReLU elementwise with default negative_slope=0.01",
          py::arg("x"),
          py::arg("negative_slope") = 0.01);
}

// Auto-register using the registry
REGISTER_BINDING(custom_ops, register_custom_ops);
