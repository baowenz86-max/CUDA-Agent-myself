// Binding for custom CUDA kernels in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void upsample_bilinear2d_launcher(
    const float* x, float* out,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    cudaStream_t stream
);

extern "C" void reflection_pad2d_launcher(
    const float* x, float* out,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    cudaStream_t stream
);

// Helper to check CUDA errors after launch
static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// PyTorch wrapper for upsample_bilinear2d
torch::Tensor upsample_bilinear2d(torch::Tensor x, double scale_factor = 2.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(scale_factor > 0.0, "scale_factor must be positive");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H_in = x.size(2);
    auto W_in = x.size(3);

    // Output dimensions same as PyTorch's floor(H * scale_factor)
    auto H_out = static_cast<int>(std::floor(H_in * scale_factor));
    auto W_out = static_cast<int>(std::floor(W_in * scale_factor));
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output dimensions must be positive");

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H_in), static_cast<int>(W_in),
        static_cast<int>(H_out), static_cast<int>(W_out),
        static_cast<float>(scale_factor),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream
    );

    check_cuda_error("upsample_bilinear2d_launcher failed");
    return output;
}

// PyTorch wrapper for reflection_pad2d
torch::Tensor reflection_pad2d(
    torch::Tensor x,
    int64_t padding_left = 1,
    int64_t padding_right = 1,
    int64_t padding_top = 1,
    int64_t padding_bottom = 1
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding_left >= 0 && padding_right >= 0 && padding_top >= 0 && padding_bottom >= 0,
                "Padding values must be non-negative");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H_in = x.size(2);
    auto W_in = x.size(3);

    auto H_out = static_cast<int>(H_in + padding_top + padding_bottom);
    auto W_out = static_cast<int>(W_in + padding_left + padding_right);
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output dimensions must be positive");

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H_in), static_cast<int>(W_in),
        static_cast<int>(H_out), static_cast<int>(W_out),
        static_cast<int>(padding_left),
        static_cast<int>(padding_right),
        static_cast<int>(padding_top),
        static_cast<int>(padding_bottom),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream
    );

    check_cuda_error("reflection_pad2d_launcher failed");
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Upsample bilinear 2D (scale_factor=2.0)",
          py::arg("x"), py::arg("scale_factor") = 2.0);
    m.def("reflection_pad2d", &reflection_pad2d,
          "Reflection pad 2D (default padding=(1,1,1,1))",
          py::arg("x"),
          py::arg("padding_left") = 1,
          py::arg("padding_right") = 1,
          py::arg("padding_top") = 1,
          py::arg("padding_bottom") = 1);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
