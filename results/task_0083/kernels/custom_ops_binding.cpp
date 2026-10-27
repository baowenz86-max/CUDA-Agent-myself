#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void interpolate_bilinear_scale_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale_factor,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride,
    cudaStream_t stream);

extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_h, int out_w,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride,
    cudaStream_t stream);

// PyTorch wrapper for bilinear upsampling with integer scale factor
torch::Tensor interpolate_bilinear_scale(torch::Tensor x, int64_t scale_factor) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));
    int out_H = H * static_cast<int>(scale_factor);
    int out_W = W * static_cast<int>(scale_factor);

    // Input may be non-contiguous, so we pass strides
    int64_t in_n_stride = x.stride(0);
    int64_t in_c_stride = x.stride(1);
    int64_t in_h_stride = x.stride(2);
    int64_t in_w_stride = x.stride(3);

    // Allocate contiguous output
    auto output = torch::empty({N, C, out_H, out_W}, x.options());
    int64_t out_n_stride = output.stride(0);
    int64_t out_c_stride = output.stride(1);
    int64_t out_h_stride = output.stride(2);
    int64_t out_w_stride = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    interpolate_bilinear_scale_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(scale_factor),
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        stream
    );

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed for interpolate_bilinear_scale");

    return output;
}

// PyTorch wrapper for adaptive average pooling
torch::Tensor adaptive_avg_pool2d(torch::Tensor x, int64_t out_h, int64_t out_w) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    TORCH_CHECK(out_h > 0 && out_w > 0, "Output spatial dimensions must be positive");

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));
    int h = static_cast<int>(out_h);
    int w = static_cast<int>(out_w);

    // Input may be non-contiguous
    int64_t in_n_stride = x.stride(0);
    int64_t in_c_stride = x.stride(1);
    int64_t in_h_stride = x.stride(2);
    int64_t in_w_stride = x.stride(3);

    // Allocate contiguous output
    auto output = torch::empty({N, C, h, w}, x.options());
    int64_t out_n_stride = output.stride(0);
    int64_t out_c_stride = output.stride(1);
    int64_t out_h_stride = output.stride(2);
    int64_t out_w_stride = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        h, w,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        stream
    );

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed for adaptive_avg_pool2d");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("interpolate_bilinear_scale", &interpolate_bilinear_scale,
          "Bilinear upsampling with integer scale factor",
          py::arg("x"), py::arg("scale_factor"));
    m.def("adaptive_avg_pool2d", &adaptive_avg_pool2d,
          "Adaptive average pooling 2D",
          py::arg("x"), py::arg("out_h"), py::arg("out_w"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
