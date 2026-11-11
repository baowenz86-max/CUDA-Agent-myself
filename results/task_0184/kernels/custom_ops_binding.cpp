#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float min_val, float max_val,
    cudaStream_t stream);

static torch::Tensor log_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");

    auto output = torch::empty_like(x);

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream
    );

    return output;
}

static torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");

    auto output = torch::empty_like(x);

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    float min_f = static_cast<float>(min_val);
    float max_f = static_cast<float>(max_val);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        min_f, max_f,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_forward", &log_softmax_forward, "LogSoftmax over dim=1 for 4D tensor (B,C,H,W)",
          py::arg("x"));
    m.def("clamp_forward", &clamp_forward, "Elementwise clamp for 4D tensor",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
