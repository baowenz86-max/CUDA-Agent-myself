#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    cudaStream_t stream
);

extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    int padding,
    double value,
    cudaStream_t stream
);

// Wrapper for softmax_forward
torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "softmax_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "softmax_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "softmax_forward: input must be 3D (batch, channels, length)");
    TORCH_CHECK(x.is_contiguous(), "softmax_forward: input must be contiguous");
    TORCH_CHECK(dim == 1, "softmax_forward: only dim=1 is supported");

    int batch = x.size(0);
    int channels = x.size(1);
    int length = x.size(2);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        channels,
        length,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for pad_forward
torch::Tensor pad_forward(torch::Tensor x, int64_t padding, double value) {
    TORCH_CHECK(x.is_cuda(), "pad_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "pad_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "pad_forward: input must be 3D (batch, channels, length)");
    TORCH_CHECK(x.is_contiguous(), "pad_forward: input must be contiguous");
    TORCH_CHECK(padding >= 0, "pad_forward: padding must be non-negative");

    int batch = x.size(0);
    int channels = x.size(1);
    int length = x.size(2);
    int output_length = length + 2 * padding;

    auto output = torch::empty({batch, channels, output_length}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pad_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        channels,
        length,
        static_cast<int>(padding),
        value,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "pad_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("softmax_forward", &softmax_forward, "Softmax along dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("pad_forward", &pad_forward, "Constant pad along last dimension",
          py::arg("x"), py::arg("padding"), py::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
