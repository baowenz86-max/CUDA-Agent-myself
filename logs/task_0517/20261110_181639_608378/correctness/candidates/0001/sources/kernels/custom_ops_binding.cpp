#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void circular_pad_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t padding,
    cudaStream_t stream);

torch::Tensor circular_pad_forward(torch::Tensor x, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must have 3 dimensions (batch, channels, length)");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto sizes = x.sizes();
    int64_t batch = sizes[0];
    int64_t channels = sizes[1];
    int64_t length = sizes[2];
    int64_t out_length = length + 2 * padding;

    auto output = torch::empty({batch, channels, out_length}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    circular_pad_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        length,
        padding,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("circular_pad_forward", &circular_pad_forward,
          "CircularPad1d forward (CUDA)",
          py::arg("x"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);