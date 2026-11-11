#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_pad_launcher(
    const float* x,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    cudaStream_t stream);

extern "C" void custom_max_dim_launcher(
    const float* x,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    cudaStream_t stream);

// PyTorch wrapper for custom_pad
torch::Tensor custom_pad_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(x.is_cuda(), "custom_pad: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_pad: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_pad: input must be 4D [N, C, H, W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "custom_pad: padding must be non-negative");

    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    auto y = torch::empty({N, C, H_out, W_out}, x.options());
    if (y.numel() == 0) return y;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_pad_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        pad_left, pad_right, pad_top, pad_bottom,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return y;
}

// PyTorch wrapper for custom_max_dim
torch::Tensor custom_max_dim_forward(torch::Tensor x, int64_t dim)
{
    TORCH_CHECK(x.is_cuda(), "custom_max_dim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_dim: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_dim: input must be 4D [N, C, H, W]");
    TORCH_CHECK(dim == 1, "custom_max_dim: only dim=1 is supported");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto y = torch::empty({N, H, W}, x.options());
    if (y.numel() == 0) return y;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_dim_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return y;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_pad", &custom_pad_forward, "ZeroPad2d (CUDA)",
          pybind11::arg("x"), pybind11::arg("pad_left"), pybind11::arg("pad_right"),
          pybind11::arg("pad_top"), pybind11::arg("pad_bottom"));
    m.def("custom_max_dim", &custom_max_dim_forward, "max over dim (CUDA)",
          pybind11::arg("x"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
