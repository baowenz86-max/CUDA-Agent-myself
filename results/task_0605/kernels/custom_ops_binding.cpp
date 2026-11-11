#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW,
    cudaStream_t stream);

extern "C" void relu_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

// Helper to compute maxpool output dimensions (same as PyTorch MaxPool2d)
static inline int64_t maxpool_output_size(
    int64_t input_size, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode) {
    int64_t numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1;
    if (ceil_mode) {
        return (numerator + stride - 1) / stride + 1;
    } else {
        return numerator / stride + 1;
    }
}

// Wrapper for custom_maxpool2d
torch::Tensor custom_maxpool2d(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool ceil_mode) {

    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor (NCHW)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must have dtype float32");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    // Ensure contiguous input (launcher assumes contiguous)
    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    // Compute output dimensions using maxpool_output_size helper
    int64_t outH = maxpool_output_size(H, kernel_h, stride_h, padding_h, dilation_h, ceil_mode);
    int64_t outW = maxpool_output_size(W, kernel_w, stride_w, padding_w, dilation_w, ceil_mode);

    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    // Allocate output tensor
    auto output = torch::empty({N, C, outH, outW}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch maxpool kernel
    maxpool2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
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
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        stream);

    return output;
}

// Wrapper for custom_relu
torch::Tensor custom_relu(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must have dtype float32");

    // Ensure contiguous input
    auto input_contig = input.contiguous();

    int64_t total = input_contig.numel();
    TORCH_CHECK(total > 0, "input must not be empty");

    // Allocate output tensor of same shape
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch relu kernel
    relu_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total),
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def(
        "custom_maxpool2d",
        &custom_maxpool2d,
        "Custom MaxPool2d forward",
        py::arg("input"),
        py::arg("kernel_h"),
        py::arg("kernel_w"),
        py::arg("stride_h"),
        py::arg("stride_w"),
        py::arg("padding_h"),
        py::arg("padding_w"),
        py::arg("dilation_h"),
        py::arg("dilation_w"),
        py::arg("ceil_mode"));

    m.def(
        "custom_relu",
        &custom_relu,
        "Custom ReLU forward",
        py::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);