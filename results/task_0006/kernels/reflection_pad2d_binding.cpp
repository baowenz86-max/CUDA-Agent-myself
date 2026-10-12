#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long total_elements,
    cudaStream_t stream
);

torch::Tensor reflection_pad2d_forward(torch::Tensor input, std::vector<int64_t> padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(padding.size() == 4, "Padding must be a 4-element vector (left, right, top, bottom)");

    int64_t pad_left = padding[0];
    int64_t pad_right = padding[1];
    int64_t pad_top = padding[2];
    int64_t pad_bottom = padding[3];

    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;

    TORCH_CHECK(out_h >= 1 && out_w >= 1, "Output size must be positive");

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    long total_elements = (long)batch * channels * out_h * out_w;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        pad_left, pad_right, pad_top, pad_bottom,
        total_elements,
        stream
    );

    return output;
}

void register_reflection_pad2d(py::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection padding 2D forward",
          py::arg("input"), py::arg("padding"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
