#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_dim1_launcher(
    float* out,
    const float* in,
    int B,
    int H,
    int W,
    long long in_stride_b,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream);

extern "C" void min_dim1_launcher(
    float* out,
    const float* in,
    int B,
    int H,
    int W,
    long long in_stride_b,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_w,
    cudaStream_t stream);

// log_softmax along dim=1
torch::Tensor log_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    int64_t B = x.size(0);
    int64_t H = x.size(1);
    int64_t W = x.size(2);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides as long long to match launcher signature
    long long in_stride_b = static_cast<long long>(x.stride(0));
    long long in_stride_h = static_cast<long long>(x.stride(1));
    long long in_stride_w = static_cast<long long>(x.stride(2));
    long long out_stride_b = static_cast<long long>(out.stride(0));
    long long out_stride_h = static_cast<long long>(out.stride(1));
    long long out_stride_w = static_cast<long long>(out.stride(2));

    log_softmax_dim1_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(W),
        in_stride_b, in_stride_h, in_stride_w,
        out_stride_b, out_stride_h, out_stride_w,
        stream);

    return out;
}

// min along dim=1
torch::Tensor min_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    int64_t B = x.size(0);
    int64_t H = x.size(1);
    int64_t W = x.size(2);

    auto out = torch::empty({B, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long in_stride_b = static_cast<long long>(x.stride(0));
    long long in_stride_h = static_cast<long long>(x.stride(1));
    long long in_stride_w = static_cast<long long>(x.stride(2));
    long long out_stride_b = static_cast<long long>(out.stride(0));
    long long out_stride_w = static_cast<long long>(out.stride(1));

    min_dim1_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(W),
        in_stride_b, in_stride_h, in_stride_w,
        out_stride_b, out_stride_w,
        stream);

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_dim1", &log_softmax_dim1, "log_softmax along dim=1", pybind11::arg("x"));
    m.def("min_dim1", &min_dim1, "min along dim=1", pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
