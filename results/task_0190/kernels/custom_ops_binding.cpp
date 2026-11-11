#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void prod_channel_keepdim_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kh, int kw, int sh, int sw, int ph, int pw,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

static torch::Tensor prod_channel_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "prod_channel_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "prod_channel_keepdim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "prod_channel_keepdim: input must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    prod_channel_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stream
    );

    return out;
}

static torch::Tensor max_pool2d(torch::Tensor x,
                                int64_t kernel_h,
                                int64_t kernel_w,
                                int64_t stride,
                                int64_t pad_h,
                                int64_t pad_w) {
    TORCH_CHECK(x.is_cuda(), "max_pool2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_pool2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "max_pool2d: input must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int kh = static_cast<int>(kernel_h);
    int kw = static_cast<int>(kernel_w);
    int sh = static_cast<int>(stride);
    int sw = static_cast<int>(stride);
    int ph = static_cast<int>(pad_h);
    int pw = static_cast<int>(pad_w);

    int H_out = (H_in + 2 * ph - kh) / sh + 1;
    int W_out = (W_in + 2 * pw - kw) / sw + 1;

    auto out = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_pool2d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out,
        kh, kw, sh, sw, ph, pw,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stream
    );

    return out;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("prod_channel_keepdim", &prod_channel_keepdim,
          "prod_channel_keepdim(x) -> Tensor\n"
          "Compute product over channel dimension (dim=1) keeping dimension size 1",
          pybind11::arg("x"));

    m.def("max_pool2d", &max_pool2d,
          "max_pool2d(x, kernel_h, kernel_w, stride, pad_h, pad_w) -> Tensor\n"
          "2D max pooling",
          pybind11::arg("x"),
          pybind11::arg("kernel_h") = 3,
          pybind11::arg("kernel_w") = 5,
          pybind11::arg("stride") = 1,
          pybind11::arg("pad_h") = 1,
          pybind11::arg("pad_w") = 2);
}

REGISTER_BINDING(custom_ops, register_custom_ops);