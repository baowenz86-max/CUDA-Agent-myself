#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    cudaStream_t stream);

extern "C" void sign_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

// Helper to check CUDA errors after launch
static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// PyTorch wrapper for unfold_forward
torch::Tensor unfold_forward(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t padding_h,
    int64_t padding_w)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilations must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");

    int64_t out_h = H + 2 * padding_h - dilation_h * (kernel_h - 1);
    int64_t out_w = W + 2 * padding_w - dilation_w * (kernel_w - 1);
    TORCH_CHECK(out_h > 0 && out_w > 0, "unfold produces empty output");

    int64_t L = out_h * out_w;
    int64_t C_times_k = C * kernel_h * kernel_w;

    // Allocate output tensor: [N, C_times_k, L]
    auto output = torch::empty({N, C_times_k, L}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    unfold_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        stream
    );
    check_cuda_error("unfold_forward launch failed");

    return output;
}

// PyTorch wrapper for sign_forward
torch::Tensor sign_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t numel = input.numel();
    auto output = torch::empty_like(input);

    if (numel > 0) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        sign_forward_launcher(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<long long>(numel),
            stream
        );
        check_cuda_error("sign_forward launch failed");
    }

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("unfold_forward", &unfold_forward,
          "Unfold forward (im2col) with stride 1",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));
    m.def("sign_forward", &sign_forward,
          "Elementwise sign",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
