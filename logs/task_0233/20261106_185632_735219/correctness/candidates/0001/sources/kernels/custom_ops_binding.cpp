// PyTorch binding for custom argmin and upsample launchers.
// No CUDA kernels here; only launcher declarations and tensor handling.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (extern "C").

extern "C" void custom_argmin_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream);

extern "C" void custom_upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    float scale_h, float scale_w,
    bool align_corners,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch.
static void cuda_check_launch_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// Custom argmin along dim=1 (C dimension).
torch::Tensor custom_argmin_forward(torch::Tensor x, int64_t dim = 1) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "custom_argmin: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_argmin: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_argmin: x must be 4D (N,C,H,W)");
    TORCH_CHECK(dim == 1, "custom_argmin: only dim=1 is supported");

    // Extract dimensions and strides (validate that strides are positive)
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int64_t stride_n = x.stride(0);
    int64_t stride_c = x.stride(1);
    int64_t stride_h = x.stride(2);
    int64_t stride_w = x.stride(3);

    // Output shape [N, H, W], dtype int64, on same device.
    auto output = torch::empty({N, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_argmin_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream
    );

    // Check for launch errors
    cuda_check_launch_error();

    return output;
}

// Custom bilinear 2D upsample with scale_factor and align_corners.
torch::Tensor custom_upsample_bilinear2d_forward(
    torch::Tensor x,
    double scale_factor = 2.0,
    bool align_corners = false) {

    // Validate input
    TORCH_CHECK(x.is_cuda(), "custom_upsample_bilinear2d: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_upsample_bilinear2d: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_upsample_bilinear2d: x must be 4D (N,C,H,W)");
    TORCH_CHECK(scale_factor > 0, "custom_upsample_bilinear2d: scale_factor must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Compute output sizes as floor(H_in*scale_factor) and floor(W_in*scale_factor).
    // Note: PyTorch uses floor for bilinear upsampling with scale_factor.
    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));

    // For scale_factor=2.0, H_out=2*H_in, etc.
    TORCH_CHECK(H_out > 0 && W_out > 0, "custom_upsample_bilinear2d: invalid output size");

    // Input strides may be non-contiguous; use actual strides from tensor.
    int64_t in_stride_n = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    // Allocate output (contiguous by default).
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Output strides for contiguous tensor.
    int64_t out_stride_n = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out,
        static_cast<float>(scale_factor), static_cast<float>(scale_factor),
        align_corners,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    // Check for launch errors
    cuda_check_launch_error();

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_argmin", &custom_argmin_forward,
          "Custom argmin along dimension 1 (C dimension).",
          py::arg("x"), py::arg("dim") = 1);

    m.def("custom_upsample_bilinear2d", &custom_upsample_bilinear2d_forward,
          "Custom bilinear 2D upsample with scale_factor and align_corners.",
          py::arg("x"), py::arg("scale_factor") = 2.0, py::arg("align_corners") = false);
}

// Auto-register binding
REGISTER_BINDING(custom_ops, register_custom_ops);
