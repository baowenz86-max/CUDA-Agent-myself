#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool1d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_width,
    cudaStream_t stream);

extern "C" void tanh_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

// MaxPool1d forward wrapper
torch::Tensor maxpool1d(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "maxpool1d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "maxpool1d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "maxpool1d: input must be 3D (batch, channels, width)");

    int batch = x.size(0);
    int channels = x.size(1);
    int width = x.size(2);

    TORCH_CHECK(kernel_size > 0, "maxpool1d: kernel_size must be positive");
    TORCH_CHECK(stride > 0, "maxpool1d: stride must be positive");
    TORCH_CHECK(padding >= 0, "maxpool1d: padding must be non-negative");

    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(output_width > 0, "maxpool1d: computed output_width is non-positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({batch, channels, output_width}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    maxpool1d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        width,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        output_width,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool1d: CUDA error after launch: ", cudaGetErrorString(err));

    return output;
}

// Tanh forward wrapper
torch::Tensor tanh(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "tanh: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "tanh: input must be float32");

    // Allocate output tensor with same shape and options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (numel as long long)
    long long numel = static_cast<long long>(x.numel());
    tanh_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tanh: CUDA error after launch: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool1d", &maxpool1d,
          "MaxPool1d forward",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));
    m.def("tanh", &tanh,
          "Tanh forward",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
