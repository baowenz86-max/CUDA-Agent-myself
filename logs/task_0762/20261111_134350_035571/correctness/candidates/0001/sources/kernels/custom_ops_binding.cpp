#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void custom_pad3d_launcher(
    const float* input,
    float* output,
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
    double value,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    cudaStream_t stream);

static torch::Tensor custom_pad3d(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    double value) {
    // Basic checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    // Output contiguous tensor, same dtype and device as input.
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides for input (in elements).
    int64_t in_stride_n = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    custom_pad3d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        value,
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        stream);

    return output;
}

void register_custom_pad3d(pybind11::module& m) {
    m.def(
        "custom_pad3d",
        &custom_pad3d,
        "Constant padding for 4D tensor",
        py::arg("x"),
        py::arg("pad_left"),
        py::arg("pad_right"),
        py::arg("pad_top"),
        py::arg("pad_bottom"),
        py::arg("value"));
}

REGISTER_BINDING(custom_pad3d, register_custom_pad3d);
