#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void custom_max_pool1d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream);

torch::Tensor custom_max_pool1d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, length)");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    int64_t numerator = L + 2 * padding - dilation * (kernel_size - 1) - 1;
    int64_t L_out;
    if (ceil_mode) {
        L_out = (numerator + stride - 1) / stride + 1;
    } else {
        L_out = numerator / stride + 1;
    }
    TORCH_CHECK(L_out >= 1, "Output length must be at least 1");

    auto output = torch::empty({B, C, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_pool1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        ceil_mode,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_max_pool1d(pybind11::module& m) {
    m.def("custom_max_pool1d", [](torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode) -> torch::Tensor {
        return custom_max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode);
    }, "Custom max pool 1d",
        pybind11::arg("x"),
        pybind11::arg("kernel_size") = 3,
        pybind11::arg("stride") = 2,
        pybind11::arg("padding") = 1,
        pybind11::arg("dilation") = 1,
        pybind11::arg("ceil_mode") = false);
}

REGISTER_BINDING(custom_max_pool1d, register_custom_max_pool1d);
