#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void constant_pad1d_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pad_left,
    int64_t pad_right,
    float value,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream);

extern "C" void interpolate_nearest_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream);

static torch::Tensor constant_pad1d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    double value) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (batch, channels, height, width)");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    int64_t out_width = width + pad_left + pad_right;
    auto output = torch::empty({batch, channels, height, out_width}, x.options());

    // Get input strides (may be non-contiguous)
    int64_t in_stride_b = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    // Output is contiguous, so strides are straightforward
    int64_t out_stride_b = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad1d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        pad_left,
        pad_right,
        static_cast<float>(value),
        in_stride_b,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad1d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

static torch::Tensor interpolate_nearest_forward(
    torch::Tensor x,
    int64_t output_height,
    int64_t output_width) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (batch, channels, height, width)");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t input_height = x.size(2);
    int64_t input_width = x.size(3);

    auto output = torch::empty({batch, channels, output_height, output_width}, x.options());

    int64_t in_stride_b = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    int64_t out_stride_b = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    interpolate_nearest_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        in_stride_b,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "interpolate_nearest kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad1d", &constant_pad1d_forward,
          "ConstantPad1d on last dimension",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("value"));
    m.def("interpolate_nearest", &interpolate_nearest_forward,
          "Nearest neighbor interpolation",
          pybind11::arg("x"),
          pybind11::arg("output_height"),
          pybind11::arg("output_width"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
