#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    double scale_factor,
    bool align_corners,
    bool recompute_scale_factor,
    cudaStream_t stream);

extern "C" void ceil_tensor_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

// Wrapper for bilinear upsampling
torch::Tensor upsample_bilinear2d(
    torch::Tensor x,
    double scale_factor,
    bool align_corners,
    bool recompute_scale_factor) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Compute output spatial dimensions (same logic as launcher)
    int H_out = static_cast<int>(H_in * scale_factor);
    int W_out = static_cast<int>(W_in * scale_factor);

    // Allocate output tensor with same options as input
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        scale_factor,
        align_corners,
        recompute_scale_factor,
        stream);

    // Check for kernel launch errors immediately (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "upsample_bilinear2d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for element-wise ceiling
torch::Tensor ceil_tensor(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    long long numel = x.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    ceil_tensor_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "ceil_tensor kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Bilinear upsampling",
          py::arg("x"),
          py::arg("scale_factor"),
          py::arg("align_corners"),
          py::arg("recompute_scale_factor") = false);
    m.def("ceil_tensor", &ceil_tensor,
          "Element-wise ceiling",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
