#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream);

extern "C" void triu_forward_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    int diagonal,
    cudaStream_t stream);

// Wrapper for log_softmax along dim=1
torch::Tensor log_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, C, H, W]");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    long long B = x.size(0);
    long long C = x.size(1);
    long long H = x.size(2);
    long long W = x.size(3);

    // Support arbitrary strides as launcher uses strides for indexing
    auto output = torch::empty_strided({B, C, H, W}, x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for triu with optional diagonal parameter
torch::Tensor triu_forward(torch::Tensor x, int64_t diagonal) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, C, H, W]");

    long long B = x.size(0);
    long long C = x.size(1);
    long long H = x.size(2);
    long long W = x.size(3);

    auto output = torch::empty_strided({B, C, H, W}, x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        static_cast<int>(diagonal),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "triu_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_forward", &log_softmax_forward,
          "log_softmax along dim=1 for 4D tensor",
          pybind11::arg("x"), pybind11::arg("dim"));
    m.def("triu_forward", &triu_forward,
          "upper triangular mask on last two dims",
          pybind11::arg("x"), pybind11::arg("diagonal") = 0);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
