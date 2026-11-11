// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// launcher declaration from custom_ops.cu
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int64_t in_stride_batch, int64_t in_stride_channel, int64_t in_stride_depth,
    int64_t in_stride_height, int64_t in_stride_width,
    int64_t out_stride_batch, int64_t out_stride_channel, int64_t out_stride_depth,
    int64_t out_stride_height, int64_t out_stride_width,
    cudaStream_t stream);

static torch::Tensor avg_pool3d_forward(
    torch::Tensor x,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width,
    int64_t padding_depth,
    int64_t padding_height,
    int64_t padding_width)
{
    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (batch, channels, depth, height, width)");

    int batch = x.size(0);
    int channels = x.size(1);
    int in_depth = x.size(2);
    int in_height = x.size(3);
    int in_width = x.size(4);

    int out_depth = (in_depth + 2 * padding_depth - kernel_depth) / stride_depth + 1;
    int out_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int out_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;
    TORCH_CHECK(out_depth > 0 && out_height > 0 && out_width > 0, "Invalid output dimensions");

    // Get input strides (in elements)
    int64_t in_stride_batch = x.stride(0);
    int64_t in_stride_channel = x.stride(1);
    int64_t in_stride_depth = x.stride(2);
    int64_t in_stride_height = x.stride(3);
    int64_t in_stride_width = x.stride(4);

    // Create output tensor with contiguous strides
    auto output_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor output = torch::empty({batch, channels, out_depth, out_height, out_width}, output_options);

    // Output strides (contiguous, in elements)
    int64_t out_stride_batch = output.stride(0);
    int64_t out_stride_channel = output.stride(1);
    int64_t out_stride_depth = output.stride(2);
    int64_t out_stride_height = output.stride(3);
    int64_t out_stride_width = output.stride(4);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        (int)kernel_depth, (int)kernel_height, (int)kernel_width,
        (int)stride_depth, (int)stride_height, (int)stride_width,
        (int)padding_depth, (int)padding_height, (int)padding_width,
        in_stride_batch, in_stride_channel, in_stride_depth,
        in_stride_height, in_stride_width,
        out_stride_batch, out_stride_channel, out_stride_depth,
        out_stride_height, out_stride_width,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool3d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

static void register_avg_pool3d_forward(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "AvgPool3d forward (CUDA)",
          py::arg("x"),
          py::arg("kernel_depth"),
          py::arg("kernel_height"),
          py::arg("kernel_width"),
          py::arg("stride_depth"),
          py::arg("stride_height"),
          py::arg("stride_width"),
          py::arg("padding_depth"),
          py::arg("padding_height"),
          py::arg("padding_width"));
}

REGISTER_BINDING(avg_pool3d_forward, register_avg_pool3d_forward);
