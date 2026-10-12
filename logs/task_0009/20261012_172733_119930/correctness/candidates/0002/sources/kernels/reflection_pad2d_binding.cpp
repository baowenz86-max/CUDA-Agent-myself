#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration, defined in kernels/reflection_pad2d.cu
extern "C" void reflection_pad2d_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_top,
    cudaStream_t stream);

// Binding function: applies 2D reflection padding to a 4D CUDA tensor
// Input: x (batch, channels, height, width)
// padding: int, symmetric padding on both dimensions (same as nn.ReflectionPad2d(padding))
static torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    // Parameter validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(padding >= 0, "Padding must be non-negative");

    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);

    const int64_t out_h = in_h + 2 * padding;
    const int64_t out_w = in_w + 2 * padding;

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(channels),
        static_cast<int>(in_h),
        static_cast<int>(in_w),
        static_cast<int>(out_h),
        static_cast<int>(out_w),
        static_cast<int>(padding),      // pad_left
        static_cast<int>(padding),      // pad_top
        stream
    );

    return output;
}

void register_reflection_pad2d(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward (CUDA)",
          pybind11::arg("input"), pybind11::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
