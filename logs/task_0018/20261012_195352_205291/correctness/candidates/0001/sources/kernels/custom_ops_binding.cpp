#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad,
    cudaStream_t stream
);

extern "C" void relu_launcher(
    float* output,
    const float* input,
    long numel,
    cudaStream_t stream
);

extern "C" void index_select_launcher(
    float* output,
    const float* input,
    long numel,
    cudaStream_t stream
);

// Wrapper for ReflectionPad2d
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t kernel_size) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_size >= 0, "kernel_size must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int pad = static_cast<int>(kernel_size);

    int out_h = H + 2 * pad;
    int out_w = W + 2 * pad;

    auto output = torch::empty({N, C, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        pad,
        stream
    );

    return output;
}

// Wrapper for ReLU
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// Wrapper for index_select (only supports full-range identity, index must be None)
torch::Tensor index_select_forward(torch::Tensor input, int64_t dim, c10::optional<torch::Tensor> index) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");
    TORCH_CHECK(!index.has_value(), "index_select_forward only supports full-range identity (index=None)");

    // For full range along dim, output shape is same as input.
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    index_select_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("kernel_size"));
    m.def("relu_forward", &relu_forward,
          "ReLU forward",
          py::arg("input"));
    m.def("index_select_forward", &index_select_forward,
          "Index select forward (full range identity)",
          py::arg("input"), py::arg("dim"), py::arg("index") = py::none());
}

REGISTER_BINDING(custom_ops, register_custom_ops);
