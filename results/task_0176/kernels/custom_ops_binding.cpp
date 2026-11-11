#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_sum_launcher(
    const float* x,
    float* out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w,
    cudaStream_t stream);

extern "C" void log_softmax_launcher(
    const float* x,
    float* y,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    long long sy_b, long long sy_h, long long sy_w,
    cudaStream_t stream);

extern "C" void sum_dim_launcher(
    const float* x,
    float* out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w,
    cudaStream_t stream);

// Helper to get strides as long long for negative or large strides
static inline long long stride_value(torch::Tensor t, int dim) {
    return static_cast<long long>(t.stride(dim));
}

// Fused log_softmax (dim=1) then sum(dim=1)
torch::Tensor log_softmax_sum_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    TORCH_CHECK(H > 0, "H must be > 0");

    auto out = torch::empty({B, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_sum_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W,
        stride_value(x, 0), stride_value(x, 1), stride_value(x, 2),
        out.stride(0), out.stride(1),
        stream
    );

    return out;
}

// log_softmax along dim=1, output same shape as input
torch::Tensor log_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    TORCH_CHECK(H > 0, "H must be > 0");

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, H, W,
        stride_value(x, 0), stride_value(x, 1), stride_value(x, 2),
        stride_value(y, 0), stride_value(y, 1), stride_value(y, 2),
        stream
    );

    return y;
}

// sum along dim=1, output shape (B, W)
torch::Tensor sum_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for 3D input");

    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);

    auto out = torch::empty({B, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W,
        stride_value(x, 0), stride_value(x, 1), stride_value(x, 2),
        out.stride(0), out.stride(1),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_sum_forward", &log_softmax_sum_forward,
          "Fused log_softmax(dim=1) + sum(dim=1)",
          pybind11::arg("x"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "log_softmax along dim=1",
          pybind11::arg("x"));
    m.def("sum_dim_forward", &sum_dim_forward,
          "sum along dim=1",
          pybind11::arg("x"),
          pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
