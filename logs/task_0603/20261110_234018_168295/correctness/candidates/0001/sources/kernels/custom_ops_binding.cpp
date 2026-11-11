#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total_elements,
                                      int64_t d1, int64_t d2, int64_t d3,
                                      int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                      cudaStream_t stream);

extern "C" void mean_dim1_forward_launcher(const float* input,
                                           float* output,
                                           int64_t batch, int64_t d1, int64_t d2, int64_t d3,
                                           int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                           int64_t out_s0, int64_t out_s1, int64_t out_s2,
                                           cudaStream_t stream);

// GELU forward wrapper
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    // Allocate output tensor with same shape as input (contiguous)
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Assuming 4D input [N, C, H, W]
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Get strides (in elements) of input
    int64_t s0 = x.stride(0);
    int64_t s1 = x.stride(1);
    int64_t s2 = x.stride(2);
    int64_t s3 = x.stride(3);

    gelu_forward_launcher(x.data_ptr<float>(),
                          output.data_ptr<float>(),
                          x.numel(),
                          C, H, W,   // d1, d2, d3 passed as C,H,W
                          s0, s1, s2, s3,
                          stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// mean_dim1 forward wrapper
torch::Tensor mean_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensor [N, H, W]
    auto output = torch::empty({N, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides (in elements) of input
    int64_t s0 = x.stride(0);
    int64_t s1 = x.stride(1);
    int64_t s2 = x.stride(2);
    int64_t s3 = x.stride(3);

    // Output strides (contiguous [N, H, W])
    int64_t out_s0 = output.stride(0);
    int64_t out_s1 = output.stride(1);
    int64_t out_s2 = output.stride(2);

    mean_dim1_forward_launcher(x.data_ptr<float>(),
                               output.data_ptr<float>(),
                               N, C, H, W,
                               s0, s1, s2, s3,
                               out_s0, out_s1, out_s2,
                               stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU forward", py::arg("x"));
    m.def("mean_dim1_forward", &mean_dim1_forward, "Mean along dim=1", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
