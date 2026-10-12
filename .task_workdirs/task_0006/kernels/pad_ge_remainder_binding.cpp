#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include "../binding_registry.h"

extern "C" void pad_ge_remainder_forward_launcher(
    int64_t* output,
    const float* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    cudaStream_t stream);

torch::Tensor pad_ge_remainder_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (batch, channels, height, width)");

    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding values must be non-negative");

    TORCH_CHECK(batch_size <= INT_MAX && channels <= INT_MAX &&
                height <= INT_MAX && width <= INT_MAX,
                "Input dimensions too large");
    TORCH_CHECK(pad_left <= INT_MAX && pad_right <= INT_MAX &&
                pad_top <= INT_MAX && pad_bottom <= INT_MAX,
                "Padding values too large");

    int64_t out_height = height + pad_top + pad_bottom;
    int64_t out_width = width + pad_left + pad_right;

    TORCH_CHECK(out_height <= INT_MAX && out_width <= INT_MAX,
                "Output dimensions too large");

    auto output = torch::empty({batch_size, channels, out_height, out_width},
                               input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pad_ge_remainder_forward_launcher(
        output.data_ptr<int64_t>(),
        input.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream);

    return output;
}

void register_pad_ge_remainder(pybind11::module& m) {
    m.def("pad_ge_remainder_forward", &pad_ge_remainder_forward,
          "Apply ReflectionPad2d, then (x >= 0), then remainder by 2\n"
          "Output dtype is int64 with values 0 or 1",
          py::arg("input"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
}

REGISTER_BINDING(pad_ge_remainder, register_pad_ge_remainder);
