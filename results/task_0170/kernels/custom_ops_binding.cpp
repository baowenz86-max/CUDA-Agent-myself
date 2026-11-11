#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avgpool2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream);

extern "C" void hardtanh_launcher(
    const float* x,
    float* out,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream);

// Wrapper for avgpool2d_forward
// Parameters:
//   x: input tensor [N, C, H, W] contiguous float32 CUDA
//   kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w: int
// Returns output tensor [N, C, outH, outW] contiguous float32 CUDA
torch::Tensor avgpool2d_forward(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "avgpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avgpool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "avgpool2d_forward: input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.is_contiguous(), "avgpool2d_forward: input must be contiguous");

    // Validate pooling parameters
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "avgpool2d_forward: kernel sizes must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "avgpool2d_forward: strides must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "avgpool2d_forward: padding must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output dimensions following floor formula
    int64_t outH = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t outW = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "avgpool2d_forward: computed output dimensions are non-positive");

    // Allocate output tensor of same dtype and device, contiguous
    auto out = torch::empty({N, C, outH, outW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointers and int dimensions
    avgpool2d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(outH),
        static_cast<int>(outW),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "avgpool2d_forward: CUDA kernel launch failed");

    return out;
}

// Wrapper for hardtanh_forward
// Parameters:
//   x: input tensor contiguous float32 CUDA
//   min_val, max_val: float scalars
// Returns output tensor shape same as x, clipped elementwise
torch::Tensor hardtanh_forward(
    torch::Tensor x,
    double min_val,
    double max_val) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "hardtanh_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardtanh_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "hardtanh_forward: input must be contiguous");

    // Allocate output tensor of same shape and options
    auto out = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with raw pointer, total elements, float min/max
    hardtanh_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(x.numel()),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hardtanh_forward: CUDA kernel launch failed");

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "AvgPool2d forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("kernel_h"),
          pybind11::arg("kernel_w"),
          pybind11::arg("stride_h"),
          pybind11::arg("stride_w"),
          pybind11::arg("padding_h"),
          pybind11::arg("padding_w"));
    m.def("hardtanh_forward", &hardtanh_forward,
          "Hardtanh forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("min_val"),
          pybind11::arg("max_val"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
