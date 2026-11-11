#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

// Declaration from custom_ops.cu
extern "C" void custom_softmax_dim1_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int length,
    cudaStream_t stream);

torch::Tensor custom_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, length)");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_softmax_dim1(pybind11::module& m) {
    m.def("custom_softmax_dim1", [](torch::Tensor x) -> torch::Tensor {
        return custom_softmax_dim1(x);
    }, "Custom softmax over dim=1", pybind11::arg("x"));
}

REGISTER_BINDING(custom_softmax_dim1, register_custom_softmax_dim1);
