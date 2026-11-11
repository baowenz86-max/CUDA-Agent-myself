#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" {
void softmax_dim1_forward(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    const int64_t* in_stride,
    const int64_t* out_stride,
    cudaStream_t stream);

void constant_pad2d_forward(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_l, int64_t pad_r, int64_t pad_t, int64_t pad_b,
    float value,
    const int64_t* in_stride,
    const int64_t* out_stride,
    cudaStream_t stream);
}

static torch::Tensor custom_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_softmax_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_softmax_dim1: input must be 4-dimensional [N, C, H, W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim1_forward(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        x.strides().data(),
        output.strides().data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_dim1: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

static torch::Tensor custom_constant_pad2d(torch::Tensor x, int64_t pad_l, int64_t pad_r, int64_t pad_t, int64_t pad_b, double value) {
    TORCH_CHECK(x.is_cuda(), "custom_constant_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_constant_pad2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_constant_pad2d: input must be 4-dimensional [N, C, H_in, W_in]");
    TORCH_CHECK(pad_l >= 0 && pad_r >= 0 && pad_t >= 0 && pad_b >= 0, "custom_constant_pad2d: padding must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = H_in + pad_t + pad_b;
    int64_t W_out = W_in + pad_l + pad_r;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad2d_forward(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H_in, W_in,
        pad_l, pad_r, pad_t, pad_b,
        static_cast<float>(value),
        x.strides().data(),
        output.strides().data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_constant_pad2d: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_dim1", &custom_softmax_dim1, "Softmax along dim=1 for [N,C,H,W] tensor", py::arg("x"));
    m.def("custom_constant_pad2d", &custom_constant_pad2d, "Constant pad 2D with explicit padding amounts",
          py::arg("x"), py::arg("pad_l"), py::arg("pad_r"), py::arg("pad_t"), py::arg("pad_b"), py::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
