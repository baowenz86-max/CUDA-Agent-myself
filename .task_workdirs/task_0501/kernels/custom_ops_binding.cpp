#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool ceil_mode,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream
);

torch::Tensor maxpool2d_forward(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool ceil_mode
) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "Kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "Stride must be positive");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "Padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "Dilation must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output spatial dimensions using floor formula
    int64_t outH = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "Output spatial dimensions must be positive");

    // Allocate contiguous output tensor
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Obtain strides for input and output (output is contiguous, but we still pass explicit strides)
    int64_t in_stride_n = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    int64_t out_stride_n = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    maxpool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        ceil_mode,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_maxpool2d(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "2D max pooling forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("kernel_h"),
          pybind11::arg("kernel_w"),
          pybind11::arg("stride_h"),
          pybind11::arg("stride_w"),
          pybind11::arg("pad_h"),
          pybind11::arg("pad_w"),
          pybind11::arg("dilation_h"),
          pybind11::arg("dilation_w"),
          pybind11::arg("ceil_mode"));
}

REGISTER_BINDING(maxpool2d, register_maxpool2d);
