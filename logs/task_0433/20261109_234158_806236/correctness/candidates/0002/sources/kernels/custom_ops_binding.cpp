// custom_ops_binding.cpp - PyTorch bindings for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernelH,
    int64_t kernelW,
    int64_t dilationH,
    int64_t dilationW,
    int64_t strideH,
    int64_t strideW,
    int64_t outH,
    int64_t outW,
    int64_t L,
    int64_t K,
    cudaStream_t stream);

// reshape launcher is a no-op in .cu; we can declare it but it is not necessary,
// we will handle reshape with tensor metadata manipulation in binding.

// PyTorch wrapper for custom_unfold_forward
torch::Tensor custom_unfold_forward(
    torch::Tensor x,
    int64_t kernel_size_h,
    int64_t kernel_size_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t stride_h,
    int64_t stride_w) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor must have 4 dimensions (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(kernel_size_h > 0 && kernel_size_w > 0, "Kernel sizes must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "Dilations must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "Strides must be positive");

    int64_t K = kernel_size_h * kernel_size_w;

    // Compute output spatial dimensions (no padding)
    int64_t outH = (H - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int64_t outW = (W - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "Kernel size too large for input spatial dims");

    int64_t L = outH * outW;

    // Output shape: [N, C*K, L]
    auto output = torch::empty({N, C * K, L}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_unfold_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size_h, kernel_size_w,
        dilation_h, dilation_w,
        stride_h, stride_w,
        outH, outW, L, K,
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for custom_reshape_3d (metadata-only, no kernel launch)
torch::Tensor custom_reshape_3d(
    torch::Tensor x,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    int64_t total = dim0 * dim1 * dim2;
    TORCH_CHECK(total == x.numel(), "Requested reshape dimensions do not match tensor size");

    // Metadata-only reshape via view
    return x.view({dim0, dim1, dim2});
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_unfold_forward", &custom_unfold_forward,
          "Unfold operation with dilation and stride",
          py::arg("x"),
          py::arg("kernel_size_h"),
          py::arg("kernel_size_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("stride_h"),
          py::arg("stride_w"));

    m.def("custom_reshape_3d", &custom_reshape_3d,
          "Reshape tensor to 3D with specified dims",
          py::arg("x"),
          py::arg("dim0"),
          py::arg("dim1"),
          py::arg("dim2"));
}

// Auto-register with the binding registry
REGISTER_BINDING(custom_ops, register_custom_ops);
