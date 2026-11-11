#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void custom_hardsigmoid_forward_launcher(
    const float* x, float* y, int64_t numel, cudaStream_t stream);

extern "C" void custom_scale_bias_forward_launcher(
    const float* x, float* y, float scale, float bias, int64_t numel,
    cudaStream_t stream);

extern "C" void custom_instance_norm_forward_launcher(
    const float* x, float* y, float* mean, float* var, int64_t N, int64_t C,
    int64_t H, int64_t W, float eps, cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t get_current_cuda_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// custom_hardsigmoid_forward
static torch::Tensor custom_hardsigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto y = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = get_current_cuda_stream();

    custom_hardsigmoid_forward_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(), numel, stream);

    return y;
}

// custom_scale_bias_forward
static torch::Tensor custom_scale_bias_forward(torch::Tensor x, double scale, double bias) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    auto y = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = get_current_cuda_stream();

    custom_scale_bias_forward_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(), static_cast<float>(scale),
        static_cast<float>(bias), numel, stream);

    return y;
}

// custom_instance_norm_forward
static torch::Tensor custom_instance_norm_forward(torch::Tensor x, double eps, double momentum) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto y = torch::empty_like(x);
    // Allocate mean and var as float32 tensors of shape (N*C) or (N, C)
    // Contract requires float32, but launcher expects float* arrays.
    auto mean = torch::empty({N * C}, x.options());
    auto var = torch::empty({N * C}, x.options());

    cudaStream_t stream = get_current_cuda_stream();

    custom_instance_norm_forward_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(), mean.data_ptr<float>(),
        var.data_ptr<float>(), N, C, H, W, static_cast<float>(eps), stream);

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_hardsigmoid_forward", &custom_hardsigmoid_forward,
          "Apply hardsigmoid elementwise: clamp(x/6 + 0.5, 0, 1)",
          pybind11::arg("x"));
    m.def("custom_scale_bias_forward", &custom_scale_bias_forward,
          "Apply scale and bias: y = x * scale + bias",
          pybind11::arg("x"), pybind11::arg("scale"), pybind11::arg("bias"));
    m.def("custom_instance_norm_forward", &custom_instance_norm_forward,
          "Apply instance normalization (no affine), y = (x - mean) / sqrt(var + eps)",
          pybind11::arg("x"), pybind11::arg("eps"), pybind11::arg("momentum"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
