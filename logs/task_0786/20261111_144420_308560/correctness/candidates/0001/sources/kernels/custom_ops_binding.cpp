// Binding for custom_mean and custom_layer_norm
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_launcher(
    const float* x,
    float* y,
    int64_t B,
    int64_t F,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_f,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w,
    cudaStream_t stream);

extern "C" void custom_layer_norm_launcher(
    const float* x,
    float* y,
    int64_t B,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w,
    float eps,
    cudaStream_t stream);

static void check_cuda_error(const char* op) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op, " CUDA error: ", cudaGetErrorString(err));
}

torch::Tensor custom_mean(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mean: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_mean: x must be 4D [B,F,H,W]");

    int64_t B = x.size(0);
    int64_t F = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(B > 0 && F > 0 && H > 0 && W > 0, "custom_mean: dimensions must be positive");

    // Output shape [B,1,H,W], contiguous tensor
    auto y = torch::empty({B, 1, H, W}, x.options());

    // Strides: input can be non-contiguous; we pass original strides
    auto sx = x.strides();
    auto sy = y.strides();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    custom_mean_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, F, H, W,
        sx[0], sx[1], sx[2], sx[3],
        sy[0], sy[1], sy[2],
        stream);

    check_cuda_error("custom_mean");
    return y;
}

torch::Tensor custom_layer_norm(torch::Tensor x, torch::Tensor normalized_shape) {
    TORCH_CHECK(x.is_cuda(), "custom_layer_norm: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_layer_norm: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_layer_norm: x must be 4D [B,1,H,W]");

    TORCH_CHECK(normalized_shape.is_cpu(), "custom_layer_norm: normalized_shape must be CPU tensor");
    TORCH_CHECK(normalized_shape.dtype() == torch::kInt64, "custom_layer_norm: normalized_shape must be int64");
    TORCH_CHECK(normalized_shape.is_contiguous(), "custom_layer_norm: normalized_shape must be contiguous");
    TORCH_CHECK(normalized_shape.numel() == 3, "custom_layer_norm: normalized_shape must have 3 elements [1,H,W]");

    auto ns_ptr = normalized_shape.data_ptr<int64_t>();
    TORCH_CHECK(ns_ptr[0] == 1, "custom_layer_norm: normalized_shape[0] must be 1");
    int64_t H = ns_ptr[1];
    int64_t W = ns_ptr[2];

    TORCH_CHECK(x.size(1) == 1, "custom_layer_norm: input dim1 must be 1");
    TORCH_CHECK(x.size(2) == H && x.size(3) == W, "custom_layer_norm: input H,W must match normalized_shape");

    int64_t B = x.size(0);
    // Output same shape as input, contiguous
    auto y = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    auto sx = x.strides();
    auto sy = y.strides();

    float eps = 1e-5f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    custom_layer_norm_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, H, W,
        sx[0], sx[2], sx[3],
        sy[0], sy[2], sy[3],
        eps,
        stream);

    check_cuda_error("custom_layer_norm");
    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean", &custom_mean, "Custom mean over dim=1 keepdim=True",
          py::arg("x"));
    m.def("custom_layer_norm", &custom_layer_norm, "Custom layer norm over [1,H,W]",
          py::arg("x"), py::arg("normalized_shape"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
