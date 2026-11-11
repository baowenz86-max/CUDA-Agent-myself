#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

namespace {

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int H_out, int W_out,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream
);

extern "C" void remainder_scalar_forward_launcher(
    const float* x,
    float* y,
    long long numel,
    float divisor,
    cudaStream_t stream
);

torch::Tensor reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N,C,H,W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t H_out = H + pad_top + pad_bottom;
    const int64_t W_out = W + pad_left + pad_right;

    // Allocate contiguous output tensor with same options as input
    auto y = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Input may be non-contiguous: pass element strides
    auto x_strides = x.strides();
    auto y_strides = y.strides();

    reflection_pad2d_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        static_cast<long long>(x_strides[0]),
        static_cast<long long>(x_strides[1]),
        static_cast<long long>(x_strides[2]),
        static_cast<long long>(x_strides[3]),
        static_cast<long long>(y_strides[0]),
        static_cast<long long>(y_strides[1]),
        static_cast<long long>(y_strides[2]),
        static_cast<long long>(y_strides[3]),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "reflection_pad2d_forward_launcher failed");
    return y;
}

torch::Tensor remainder_scalar_forward(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    auto y = torch::empty_like(x);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    remainder_scalar_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        static_cast<float>(divisor),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "remainder_scalar_forward_launcher failed");
    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));

    m.def("remainder_scalar_forward", &remainder_scalar_forward,
          "Remainder with scalar (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("divisor"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
