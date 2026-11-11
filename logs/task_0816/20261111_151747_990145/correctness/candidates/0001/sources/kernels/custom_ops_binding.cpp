#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_upsample_bilinear2d_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    float scale_factor,
    bool align_corners,
    cudaStream_t stream);

extern "C" void custom_mean_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Wrapper for custom_upsample_bilinear2d
torch::Tensor custom_upsample_bilinear2d(
    torch::Tensor x,
    double scale_factor,
    bool align_corners) {
    // Check tensor properties
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Compute output dimensions: floor(H * scale_factor)
    int H_out = static_cast<int>(H * scale_factor);
    int W_out = static_cast<int>(W * scale_factor);

    // Allocate output tensor
    auto y = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(scale_factor),
        align_corners,
        stream);

    return y;
}

// Wrapper for custom_mean_dim1_keepdim
torch::Tensor custom_mean_dim1_keepdim(torch::Tensor x) {
    // Check tensor properties
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor [N, 1, H, W]
    auto y = torch::empty({N, 1, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_mean_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        stream);

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_bilinear2d", &custom_upsample_bilinear2d,
          "CUDA bilinear upsampling (align_corners=false)",
          py::arg("x"), py::arg("scale_factor"), py::arg("align_corners"));
    m.def("custom_mean_dim1_keepdim", &custom_mean_dim1_keepdim,
          "CUDA mean along dim=1 with keepdim=True",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
