#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void affine_grid_scaled_launcher(
    const float* theta,
    float* output,
    int64_t N, int64_t H, int64_t W,
    int64_t theta_stride_n, int64_t theta_stride_r, int64_t theta_stride_c,
    bool align_corners,
    float scale,
    cudaStream_t stream);

// PyTorch wrapper for affine grid generation + scale
torch::Tensor affine_grid_scaled(
    torch::Tensor theta,
    int64_t size_h,
    int64_t size_w,
    bool align_corners,
    double scale)
{
    // Check input tensor
    TORCH_CHECK(theta.is_cuda(), "theta must be a CUDA tensor");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta must be float32");
    TORCH_CHECK(theta.dim() == 3, "theta must have shape [N, 2, 3]");
    TORCH_CHECK(theta.size(1) == 2 && theta.size(2) == 3, "theta must have shape [N, 2, 3]");

    int64_t N = theta.size(0);
    TORCH_CHECK(N > 0, "theta batch size must be positive");
    TORCH_CHECK(size_h > 0, "size_h must be positive");
    TORCH_CHECK(size_w > 0, "size_w must be positive");

    // Output tensor: [N, size_h, size_w, 2]
    auto output = torch::empty({N, size_h, size_w, 2}, theta.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // theta may be non-contiguous; pass strides
    int64_t stride_n = theta.stride(0);
    int64_t stride_r = theta.stride(1);
    int64_t stride_c = theta.stride(2);

    // Call CUDA launcher
    affine_grid_scaled_launcher(
        theta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, size_h, size_w,
        stride_n, stride_r, stride_c,
        align_corners,
        static_cast<float>(scale),
        stream);

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

// Registration function
void register_affine_grid_scaled(pybind11::module& m) {
    m.def("affine_grid_scaled", &affine_grid_scaled,
          "Generate affine grid and scale by scalar",
          pybind11::arg("theta"),
          pybind11::arg("size_h"),
          pybind11::arg("size_w"),
          pybind11::arg("align_corners"),
          pybind11::arg("scale"));
}

REGISTER_BINDING(affine_grid_scaled, register_affine_grid_scaled);
