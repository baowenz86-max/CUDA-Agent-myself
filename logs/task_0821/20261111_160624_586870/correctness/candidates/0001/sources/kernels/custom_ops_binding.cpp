#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int B, int C, int inH, int inW,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int in_stride_b, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_b, int out_stride_c, int out_stride_h, int out_stride_w,
    cudaStream_t stream
);

extern "C" void argsort_dim1_launcher(
    const float* input,
    long long* output,
    int B, int C, int H, int W,
    int in_stride_b, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_b, int out_stride_c, int out_stride_h, int out_stride_w,
    cudaStream_t stream
);

torch::Tensor reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Extract dimensions
    int B = x.size(0);
    int C = x.size(1);
    int inH = x.size(2);
    int inW = x.size(3);

    // Compute output dimensions
    int outH = inH + pad_top + pad_bottom;
    int outW = inW + pad_left + pad_right;

    // Allocate output tensor (contiguous NCHW)
    auto output = torch::empty({B, C, outH, outW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides (contiguous, but computed for safety)
    int in_stride_b = x.stride(0);
    int in_stride_c = x.stride(1);
    int in_stride_h = x.stride(2);
    int in_stride_w = x.stride(3);

    int out_stride_b = output.stride(0);
    int out_stride_c = output.stride(1);
    int out_stride_h = output.stride(2);
    int out_stride_w = output.stride(3);

    // Call launcher
    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, inH, inW,
        static_cast<int>(pad_left), static_cast<int>(pad_right),
        static_cast<int>(pad_top), static_cast<int>(pad_bottom),
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in reflection_pad2d_launcher");

    return output;
}

torch::Tensor argsort_dim1_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output as int64 (long)
    auto output = torch::empty({B, C, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int in_stride_b = x.stride(0);
    int in_stride_c = x.stride(1);
    int in_stride_h = x.stride(2);
    int in_stride_w = x.stride(3);

    int out_stride_b = output.stride(0);
    int out_stride_c = output.stride(1);
    int out_stride_h = output.stride(2);
    int out_stride_w = output.stride(3);

    argsort_dim1_launcher(
        x.data_ptr<float>(),
        static_cast<long long*>(output.data_ptr<int64_t>()),
        B, C, H, W,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in argsort_dim1_launcher");

    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));

    m.def("argsort_dim1_forward", &argsort_dim1_forward,
          "Argsort along dim=1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
