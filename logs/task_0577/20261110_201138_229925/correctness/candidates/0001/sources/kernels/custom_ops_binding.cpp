// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher functions from custom_ops.cu
extern "C" void custom_abs_launcher(
    const float* x,
    float* y,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_layer_norm_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    double eps,
    int64_t batch_size,
    int64_t norm_size,
    int64_t spatial_stride_b,
    int64_t spatial_stride_c,
    int64_t spatial_stride_h,
    int64_t spatial_stride_w,
    int64_t weight_stride_c,
    int64_t weight_stride_h,
    int64_t weight_stride_w,
    bool has_affine,
    cudaStream_t stream);

// PyTorch wrapper for custom_abs
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_abs: input must be contiguous");

    auto y = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs: CUDA error: ", cudaGetErrorString(err));

    return y;
}

// PyTorch wrapper for custom_layer_norm
torch::Tensor custom_layer_norm(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "custom_layer_norm: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_layer_norm: x must be float32");
    TORCH_CHECK(x.dim() >= 3, "custom_layer_norm: x must have at least 3 dimensions");

    if (weight.has_value()) {
        TORCH_CHECK(weight->is_cuda(), "custom_layer_norm: weight must be a CUDA tensor");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "custom_layer_norm: weight must be float32");
        TORCH_CHECK(weight->is_contiguous(), "custom_layer_norm: weight must be contiguous");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "custom_layer_norm: bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "custom_layer_norm: bias must be float32");
        TORCH_CHECK(bias->is_contiguous(), "custom_layer_norm: bias must be contiguous");
    }

    int64_t batch_size = x.size(0);
    int64_t norm_size = 1;
    for (int64_t i = 1; i < x.dim(); ++i) {
        norm_size *= x.size(i);
    }

    auto y = torch::empty_like(x);

    // Get strides (assume input may be non-contiguous but we handle in kernel via spatial strides)
    int64_t spatial_stride_b = x.stride(0);
    int64_t spatial_stride_c = x.stride(1);
    int64_t spatial_stride_h = x.stride(2);
    int64_t spatial_stride_w = x.stride(3);

    // weight/bias are expected shape (C,H,W) and we treat them as contiguous
    int64_t weight_stride_c = 0, weight_stride_h = 0, weight_stride_w = 0;
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    bool has_affine = false;

    if (weight.has_value() && bias.has_value()) {
        TORCH_CHECK(weight->numel() == norm_size, "custom_layer_norm: weight size mismatch");
        TORCH_CHECK(bias->numel() == norm_size, "custom_layer_norm: bias size mismatch");
        // weight and bias are contiguous with shape (C,H,W), compute strides
        weight_stride_c = weight->stride(0);
        weight_stride_h = weight->stride(1);
        weight_stride_w = weight->stride(2);
        weight_ptr = weight->data_ptr<float>();
        bias_ptr = bias->data_ptr<float>();
        has_affine = true;
    } else if (weight.has_value() || bias.has_value()) {
        TORCH_CHECK(false, "custom_layer_norm: weight and bias must both be provided or both None");
    } else {
        // no affine, set weight/bias pointers to nullptr, strides irrelevant
        weight_ptr = nullptr;
        bias_ptr = nullptr;
        has_affine = false;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_layer_norm_launcher(
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        y.data_ptr<float>(),
        eps,
        batch_size,
        norm_size,
        spatial_stride_b,
        spatial_stride_c,
        spatial_stride_h,
        spatial_stride_w,
        weight_stride_c,
        weight_stride_h,
        weight_stride_w,
        has_affine,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_layer_norm: CUDA error: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Custom absolute value",
          py::arg("x"));
    m.def("custom_layer_norm", &custom_layer_norm, "Custom layer normalization",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
