#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_height,
    long long in_stride_width,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_height,
    long long out_stride_width,
    cudaStream_t stream
);

namespace py = pybind11;

// Binding wrapper
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");

    int pad_left = static_cast<int>(padding);
    int pad_right = static_cast<int>(padding);
    int pad_top = static_cast<int>(padding);
    int pad_bottom = static_cast<int>(padding);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_height = in_height + pad_top + pad_bottom;
    int out_width = in_width + pad_left + pad_right;

    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        stream
    );

    return output;
}

void register_reflection_pad2d(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
