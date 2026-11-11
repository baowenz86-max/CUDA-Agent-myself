#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void selu_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream);

extern "C" void cumsum0_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream);

static torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    int rows = x.size(0);
    int cols = x.size(1);
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        stream);

    return output;
}

static torch::Tensor cumsum0_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    int rows = x.size(0);
    int cols = x.size(1);
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cumsum0_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("selu_forward", &selu_forward, "SELU activation forward");
    m.def("cumsum0_forward", &cumsum0_forward, "Cumulative sum along dimension 0");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
