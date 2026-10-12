// PyTorch binding for reflection_pad2d custom op
#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from reflection_pad2d.cu
extern "C" void launch_reflection_pad2d(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_h, int pad_w,
    cudaStream_t stream);

torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(padding >= 0, "Padding must be non-negative");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));
    int pad_h = static_cast<int>(padding);
    int pad_w = static_cast<int>(padding);
    int out_H = H + 2 * pad_h;
    int out_W = W + 2 * pad_w;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_reflection_pad2d(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        pad_h, pad_w,
        stream);

    return output;
}

void register_reflection_pad2d(py::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
