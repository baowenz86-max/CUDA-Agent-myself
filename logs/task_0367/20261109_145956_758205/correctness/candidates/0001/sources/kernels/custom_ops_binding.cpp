#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    long long in_stride_n,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_n,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream
);

// Wrapper function for PyTorch
static torch::Tensor custom_adaptive_avg_pool2d_forward(torch::Tensor x, int64_t output_size) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    int batch_size = x.size(0);
    int channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    // Output shape: [N, C, output_size, output_size]
    auto output = torch::empty({batch_size, channels, output_size, output_size}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Input strides (elements, can be non-contiguous)
    long long in_stride_n = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_h = x.stride(2);
    long long in_stride_w = x.stride(3);

    // Output is contiguous, compute strides
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    // Call CUDA launcher
    custom_adaptive_avg_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        static_cast<int>(output_size),
        static_cast<int>(output_size),
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// Registration function
static void register_custom_adaptive_avg_pool2d(pybind11::module& m) {
    m.def(
        "custom_adaptive_avg_pool2d",
        &custom_adaptive_avg_pool2d_forward,
        "Custom AdaptiveAvgPool2d forward",
        pybind11::arg("x"),
        pybind11::arg("output_size") = 5
    );
}

// Auto-register
REGISTER_BINDING(custom_adaptive_avg_pool2d, register_custom_adaptive_avg_pool2d);
