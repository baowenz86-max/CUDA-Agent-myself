#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t launch_reduce_mean_channel(
    const float* x,
    float* y,
    long long B,
    long long C,
    long long H,
    long long W,
    long long stride_b,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

extern "C" cudaError_t launch_add_scalar(
    const float* x,
    float* y,
    long long total_elements,
    float scalar,
    cudaStream_t stream
);

// Wrapper for custom_reduce_mean_channel
torch::Tensor custom_reduce_mean_channel(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional [B, C, H, W]");
    TORCH_CHECK(x.size(1) == 3, "Channel dimension must be 3");

    const long long B = x.size(0);
    const long long C = x.size(1);
    const long long H = x.size(2);
    const long long W = x.size(3);

    // Output tensor shape [B, H, W] contiguous
    auto y = torch::empty({B, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = launch_reduce_mean_channel(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_reduce_mean_channel failed with error: ", cudaGetErrorString(err));

    return y;
}

// Wrapper for custom_add_scalar
torch::Tensor custom_add_scalar(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = launch_add_scalar(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x.numel(),
        static_cast<float>(scalar),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_add_scalar failed with error: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_reduce_mean_channel", &custom_reduce_mean_channel,
          "Compute mean over channel dimension (C=3)",
          pybind11::arg("x"));
    m.def("custom_add_scalar", &custom_add_scalar,
          "Add scalar to every element",
          pybind11::arg("x"), pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
