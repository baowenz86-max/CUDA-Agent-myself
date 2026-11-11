#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
);

extern "C" void tanh_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
);

// Average pooling forward binding
torch::Tensor avgpool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");

    int batch = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_height = (in_height + 2 * static_cast<int>(padding) - static_cast<int>(kernel_size)) / static_cast<int>(stride) + 1;
    int out_width = (in_width + 2 * static_cast<int>(padding) - static_cast<int>(kernel_size)) / static_cast<int>(stride) + 1;

    auto output = torch::empty({batch, channels, out_height, out_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avgpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_height,
        in_width,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    return output;
}

// Tanh forward binding
torch::Tensor tanh_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "Average pooling 2D forward",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));
    m.def("tanh_forward", &tanh_forward,
          "Tanh activation forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
