#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void lp_pool2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int norm_type,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    bool ceil_mode,
    cudaStream_t stream
);

// PyTorch wrapper for lp_pool2d_forward
torch::Tensor lp_pool2d_forward(
    torch::Tensor input,
    int64_t norm_type,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding_h,
    int64_t padding_w,
    bool ceil_mode
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Extract dimensions
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Compute output dimensions (matching nn.LPPool2d with same padding logic)
    int64_t out_h, out_w;
    if (ceil_mode) {
        out_h = (H + 2 * padding_h - kernel_h + stride - 1) / stride + 1;
        out_w = (W + 2 * padding_w - kernel_w + stride - 1) / stride + 1;
    } else {
        out_h = (H + 2 * padding_h - kernel_h) / stride + 1;
        out_w = (W + 2 * padding_w - kernel_w) / stride + 1;
    }

    // Handle non-positive output sizes (shouldn't happen for valid inputs)
    TORCH_CHECK(out_h > 0 && out_w > 0, "Output size is non-positive");

    // Allocate output tensor with same options as input
    auto output = torch::empty({N, C, out_h, out_w}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    lp_pool2d_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(norm_type),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(stride),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        ceil_mode,
        stream
    );

    return output;
}

// Registration function
void register_lp_pool2d(pybind11::module& m) {
    m.def(
        "lp_pool2d_forward",
        &lp_pool2d_forward,
        "LP Pooling 2D forward (norm_type=2)",
        py::arg("input"),
        py::arg("norm_type"),
        py::arg("kernel_h"),
        py::arg("kernel_w"),
        py::arg("stride"),
        py::arg("padding_h"),
        py::arg("padding_w"),
        py::arg("ceil_mode")
    );
}

// Auto-register
REGISTER_BINDING(lp_pool2d, register_lp_pool2d);
