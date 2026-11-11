// kernels/custom_ops_binding.cpp
// PyTorch bindings for custom_unfold and custom_pad1d CUDA kernels.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void custom_unfold_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_size, int dilation, int pad_h, int pad_w, int stride,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

extern "C" void custom_pad1d_launcher(
    const float* input,
    float* output,
    int N, int C, int L,
    int pad_left, int pad_right,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

// Helper to print or throw error on CUDA launch failure
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// custom_unfold forward: x (float32 CUDA 4D) -> out (float32 CUDA 3D)
torch::Tensor custom_unfold_forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t dilation,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "custom_unfold: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_unfold: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_unfold: input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output spatial dimensions exactly as nn.Unfold
    int64_t out_h = (H + 2 * padding_h - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding_w - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t L = out_h * out_w;
    int64_t C_out = C * kernel_size * kernel_size;

    // Allocate output tensor [N, C_out, L], contiguous by default
    auto output = torch::empty({N, C_out, L}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Retrieve strides (input may be non-contiguous)
    auto x_strides = x.strides();
    auto out_strides = output.strides();

    // Call launcher with mismatched types casted (launcher expects long long)
    custom_unfold_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(kernel_size),
        static_cast<int>(dilation),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(stride),
        static_cast<long long>(x_strides[0]),
        static_cast<long long>(x_strides[1]),
        static_cast<long long>(x_strides[2]),
        static_cast<long long>(x_strides[3]),
        static_cast<long long>(out_strides[0]),
        static_cast<long long>(out_strides[1]),
        static_cast<long long>(out_strides[2]),
        stream
    );
    check_cuda_error("custom_unfold_launcher failed");

    return output;
}

// custom_pad1d forward: x (float32 CUDA 3D) -> out (float32 CUDA 3D)
torch::Tensor custom_pad1d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "custom_pad1d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_pad1d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_pad1d: input must be 3D (N, C, L)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    // Output length
    int64_t L_out = L + pad_left + pad_right;

    // Allocate output tensor [N, C, L_out], contiguous by default
    auto output = torch::empty({N, C, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto x_strides = x.strides();
    auto out_strides = output.strides();

    custom_pad1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<long long>(x_strides[0]),
        static_cast<long long>(x_strides[1]),
        static_cast<long long>(x_strides[2]),
        static_cast<long long>(out_strides[0]),
        static_cast<long long>(out_strides[1]),
        static_cast<long long>(out_strides[2]),
        stream
    );
    check_cuda_error("custom_pad1d_launcher failed");

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_unfold", &custom_unfold_forward,
          "Im2col style unfold (float32 CUDA)",
          pybind11::arg("x"),
          pybind11::arg("kernel_size"),
          pybind11::arg("dilation"),
          pybind11::arg("padding_h"),
          pybind11::arg("padding_w"),
          pybind11::arg("stride"));
    m.def("custom_pad1d", &custom_pad1d_forward,
          "Constant pad last dimension (float32 CUDA)",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
