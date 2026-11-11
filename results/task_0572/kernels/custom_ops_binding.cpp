#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    cudaStream_t stream);

// Wrapper for custom_abs_forward
torch::Tensor custom_abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_abs_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_abs_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Wrapper for adaptive_avg_pool2d_forward
torch::Tensor adaptive_avg_pool2d_forward(torch::Tensor x, int64_t out_h, int64_t out_w) {
    TORCH_CHECK(x.is_cuda(), "adaptive_avg_pool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "adaptive_avg_pool2d_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "adaptive_avg_pool2d_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "adaptive_avg_pool2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(out_h > 0 && out_w > 0, "adaptive_avg_pool2d_forward: output dimensions must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({N, C, out_h, out_w}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        out_h, out_w,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("adaptive_avg_pool2d_forward", &adaptive_avg_pool2d_forward, "2D adaptive average pooling",
          py::arg("x"), py::arg("out_h"), py::arg("out_w"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
