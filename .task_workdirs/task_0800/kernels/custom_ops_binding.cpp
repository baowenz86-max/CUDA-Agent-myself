#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher functions from custom_ops.cu
extern "C" void isin_forward_launcher(
    const float* x,
    const float* values,
    bool* out,
    int64_t numel,
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    int64_t values_count,
    cudaStream_t stream);

extern "C" void maxpool2d_forward_launcher(
    const float* x,
    float* y,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream);

// Wrapper for isin_forward
torch::Tensor isin_forward(torch::Tensor x, torch::Tensor values) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (batch, channels, height, width)");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");

    // values must be contiguous for the launcher
    auto values_contig = values.contiguous();

    // Allocate output tensor with same shape as x, dtype bool
    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kBool));

    // Get dimensions and strides of x (may be non-contiguous)
    int64_t n = x.size(0);
    int64_t c = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    int64_t stride_n = x.stride(0);
    int64_t stride_c = x.stride(1);
    int64_t stride_h = x.stride(2);
    int64_t stride_w = x.stride(3);
    int64_t numel = x.numel();
    int64_t values_count = values_contig.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    isin_forward_launcher(
        x.data_ptr<float>(),
        values_contig.data_ptr<float>(),
        out.data_ptr<bool>(),
        numel,
        n, c, h, w,
        stride_n, stride_c, stride_h, stride_w,
        values_count,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in isin_forward_launcher: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for maxpool2d_forward
torch::Tensor maxpool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (batch, channels, height, width)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous for cuDNN");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    // Compute output dimensions using floor formula
    int64_t out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int64_t out_width  = (width  + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(out_height > 0 && out_width > 0, "Invalid pooling parameters lead to non-positive output size");

    // Allocate output tensor
    auto out = torch::empty({batch, channels, out_height, out_width}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    maxpool2d_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, channels, height, width,
        kernel_size, stride, padding,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in maxpool2d_forward_launcher: ", cudaGetErrorString(err));

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("isin_forward", &isin_forward, "Compute mask of elements in x that are in values",
          py::arg("x"), py::arg("values"));
    m.def("maxpool2d_forward", &maxpool2d_forward, "2D max pooling using cuDNN",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
