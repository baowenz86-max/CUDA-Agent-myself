#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t custom_mean_dim1_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long batch_stride, long long channel_stride,
    long long height_stride, long long width_stride,
    cudaStream_t stream);

extern "C" cudaError_t custom_leaky_relu_launcher(
    const float* x,
    float* out,
    long long numel,
    double negative_slope,
    cudaStream_t stream);

// custom_mean_dim1: mean over dim=1 with keepdim=True
torch::Tensor custom_mean_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C, H, W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Compute strides (long long to match launcher)
    long long batch_stride = x.stride(0);
    long long channel_stride = x.stride(1);
    long long height_stride = x.stride(2);
    long long width_stride = x.stride(3);

    auto out = torch::empty({N, 1, H, W}, x.options());
    // Ensure contiguous output
    out = out.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = custom_mean_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        batch_stride, channel_stride, height_stride, width_stride,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "custom_mean_dim1_launcher failed with error ", cudaGetErrorString(err));

    return out;
}

// custom_leaky_relu: elementwise leaky ReLU
torch::Tensor custom_leaky_relu_forward(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Output shape same as input; use empty_like to preserve layout
    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = custom_leaky_relu_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel(),
        negative_slope,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "custom_leaky_relu_launcher failed with error ", cudaGetErrorString(err));

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim1", &custom_mean_dim1_forward, "Mean over dim=1 keepdim=True",
          py::arg("x"));
    m.def("custom_leaky_relu", &custom_leaky_relu_forward, "LeakyReLU elementwise",
          py::arg("x"), py::arg("negative_slope"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
