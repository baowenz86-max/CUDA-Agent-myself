#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream);

extern "C" void logical_and_gt_lt_launcher(
    const float* input,
    bool* output,
    long long numel,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    float lower,
    float upper,
    cudaStream_t stream);

// Wrapper for upsample_bilinear2d
static torch::Tensor custom_upsample_bilinear2d(
    torch::Tensor x,
    double scale_factor,
    bool align_corners) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Compute output dimensions: floor(H_in * scale_factor), floor(W_in * scale_factor)
    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));

    // Output is contiguous, same dtype/device
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass input strides as long long (they are positive for any layout)
    long long input_stride_n = x.stride(0);
    long long input_stride_c = x.stride(1);
    long long input_stride_h = x.stride(2);
    long long input_stride_w = x.stride(3);

    upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        static_cast<float>(scale_factor),
        align_corners,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "upsample_bilinear2d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for logical_and_gt_lt
static torch::Tensor custom_logical_and_gt_lt(
    torch::Tensor x,
    double lower,
    double upper) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D as per current contract");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long numel = x.numel();
    long long input_stride_n = x.stride(0);
    long long input_stride_c = x.stride(1);
    long long input_stride_h = x.stride(2);
    long long input_stride_w = x.stride(3);

    logical_and_gt_lt_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        numel,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        static_cast<float>(lower),
        static_cast<float>(upper),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "logical_and_gt_lt kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_bilinear2d", &custom_upsample_bilinear2d,
          "Upsample bilinear2d custom op",
          pybind11::arg("x"),
          pybind11::arg("scale_factor"),
          pybind11::arg("align_corners"));
    m.def("custom_logical_and_gt_lt", &custom_logical_and_gt_lt,
          "Logical AND of x > lower and x < upper",
          pybind11::arg("x"),
          pybind11::arg("lower"),
          pybind11::arg("upper"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
