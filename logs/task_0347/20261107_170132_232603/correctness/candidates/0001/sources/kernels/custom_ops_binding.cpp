// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int inp_sN, int inp_sC, int inp_sH, int inp_sW,
    cudaStream_t stream);

// PyTorch wrapper for unfold (im2col)
torch::Tensor unfold(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Ensure input is contiguous; if not, we'll handle strides, but for simplicity
    // we'll require contiguity? No, launcher supports strides.
    // We'll pass strides as is.
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output dimensions
    int64_t out_h = H - dilation_h * (kernel_h - 1);
    int64_t out_w = W - dilation_w * (kernel_w - 1);
    TORCH_CHECK(out_h > 0 && out_w > 0, "Kernel size too large for input");

    int64_t K = kernel_h * kernel_w;
    int64_t L = out_h * out_w;

    // Allocate output tensor: contiguous [N, C*K, L]
    auto output = torch::empty({N, C * K, L}, x.options());

    // Get input strides
    auto strides = x.strides();
    int64_t inp_sN = strides[0];
    int64_t inp_sC = strides[1];
    int64_t inp_sH = strides[2];
    int64_t inp_sW = strides[3];

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with int64_t converted to int (should be safe for practical sizes)
    unfold_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(inp_sN),
        static_cast<int>(inp_sC),
        static_cast<int>(inp_sH),
        static_cast<int>(inp_sW),
        stream);

    return output;
}

// Registration
void register_unfold(pybind11::module& m) {
    m.def("unfold", &unfold,
          "Unfold (im2col) with stride=1, padding=0",
          pybind11::arg("x"),
          pybind11::arg("kernel_h"),
          pybind11::arg("kernel_w"),
          pybind11::arg("dilation_h"),
          pybind11::arg("dilation_w"));
}

REGISTER_BINDING(unfold, register_unfold);
