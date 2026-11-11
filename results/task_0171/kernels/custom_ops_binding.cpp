#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_dim1_launcher(
    const float* input,
    float* output,
    int B,
    int N,
    int C,
    long long in_stride_b,
    long long in_stride_n,
    long long in_stride_c,
    long long out_stride_b,
    long long out_stride_n,
    long long out_stride_c,
    cudaStream_t stream);

extern "C" void elu_launcher(
    const float* input,
    float* output,
    long long total_elements,
    int B,
    int N,
    int C,
    long long input_stride_b,
    long long input_stride_n,
    long long input_stride_c,
    long long output_stride_b,
    long long output_stride_n,
    long long output_stride_c,
    cudaStream_t stream);

static torch::Tensor log_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.dim() == 3, "input must be 3D (B, N, C)");

    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    // Allocate output with same shape and strides as input (empty_like preserves strides)
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<long long>(x.stride(0)),
        static_cast<long long>(x.stride(1)),
        static_cast<long long>(x.stride(2)),
        static_cast<long long>(output.stride(0)),
        static_cast<long long>(output.stride(1)),
        static_cast<long long>(output.stride(2)),
        stream
    );

    return output;
}

static torch::Tensor elu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.dim() == 3, "input must be 3D (B, N, C)");

    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    elu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        static_cast<int>(B),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<long long>(x.stride(0)),
        static_cast<long long>(x.stride(1)),
        static_cast<long long>(x.stride(2)),
        static_cast<long long>(output.stride(0)),
        static_cast<long long>(output.stride(1)),
        static_cast<long long>(output.stride(2)),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_dim1", &log_softmax_dim1, "log_softmax over dim=1", py::arg("x"));
    m.def("elu", &elu, "ELU activation", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);