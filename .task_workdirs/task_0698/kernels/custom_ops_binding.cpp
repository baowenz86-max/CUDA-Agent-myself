#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c,
    long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_0,
    long long output_stride_h, long long output_stride_w,
    cudaStream_t stream);

extern "C" void relu_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream);

// Binding for mean_dim1_keepdim
torch::Tensor mean_dim1_keepdim_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C, H, W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape [N, 1, H, W] contiguous
    auto y = torch::empty({N, 1, H, W}, x.options());

    // Get strides (long long as expected by launcher)
    long long input_stride_n = x.stride(0);
    long long input_stride_c = x.stride(1);
    long long input_stride_h = x.stride(2);
    long long input_stride_w = x.stride(3);
    long long output_stride_n = y.stride(0);
    long long output_stride_0 = y.stride(1);
    long long output_stride_h = y.stride(2);
    long long output_stride_w = y.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_0, output_stride_h, output_stride_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim1_keepdim kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Binding for relu
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto y = torch::empty_like(x);
    long long numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim1_keepdim", &mean_dim1_keepdim_forward, "Mean over dim 1 keeping dimension", py::arg("x"));
    m.def("relu", &relu_forward, "ReLU activation", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
