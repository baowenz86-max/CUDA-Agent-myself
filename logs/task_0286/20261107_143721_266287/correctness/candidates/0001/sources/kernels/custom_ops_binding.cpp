// Binding for upsample_nearest2d and relu custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
// upsample_nearest2d_launcher:
//   const float* x, float* out, int N, int C, int H, int W, int scale_factor,
//   long long n_stride, long long c_stride, long long h_stride, long long w_stride,
//   cudaStream_t stream
extern "C" void upsample_nearest2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W, int scale_factor,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    cudaStream_t stream);

// relu_launcher:
//   const float* x, float* out, long long numel,
//   int N, int C, int H, int W,
//   long long n_stride, long long c_stride, long long h_stride, long long w_stride,
//   cudaStream_t stream
extern "C" void relu_launcher(
    const float* x,
    float* out,
    long long numel,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    cudaStream_t stream);

// Utility to check 4D NCHW tensor and extract dims
static void check_input_4d(const torch::Tensor& x, const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.dim() == 4, op_name, ": input must be 4D (N, C, H, W)");
}

// Convert tensor to contiguous if not already, then compute strides of contiguous layout
static void get_contiguous_strides(
    const torch::Tensor& t,
    int N, int C, int H, int W,
    long long& n_stride, long long& c_stride, long long& h_stride, long long& w_stride) {
    // For a contiguous NCHW tensor, strides are (C*H*W, H*W, W, 1)
    n_stride = static_cast<long long>(C) * H * W;
    c_stride = static_cast<long long>(H) * W;
    h_stride = W;
    w_stride = 1;
}

// upsample_nearest2d binding
torch::Tensor upsample_nearest2d(torch::Tensor x, int64_t scale_factor) {
    check_input_4d(x, "upsample_nearest2d");
    TORCH_CHECK(scale_factor > 0, "upsample_nearest2d: scale_factor must be positive");
    TORCH_CHECK(x.is_contiguous(), "upsample_nearest2d: input must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int out_h = H * static_cast<int>(scale_factor);
    int out_w = W * static_cast<int>(scale_factor);

    auto out = torch::empty({N, C, out_h, out_w}, x.options());

    // Since we require contiguous input, use contiguous strides
    long long n_stride, c_stride, h_stride, w_stride;
    get_contiguous_strides(x, N, C, H, W, n_stride, c_stride, h_stride, w_stride);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_nearest2d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, static_cast<int>(scale_factor),
        n_stride, c_stride, h_stride, w_stride,
        stream);

    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// relu binding
torch::Tensor relu(torch::Tensor x) {
    check_input_4d(x, "relu");
    TORCH_CHECK(x.is_contiguous(), "relu: input must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    long long numel = static_cast<long long>(N) * C * H * W;

    auto out = torch::empty_like(x);

    long long n_stride, c_stride, h_stride, w_stride;
    get_contiguous_strides(x, N, C, H, W, n_stride, c_stride, h_stride, w_stride);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        N, C, H, W,
        n_stride, c_stride, h_stride, w_stride,
        stream);

    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_nearest2d", &upsample_nearest2d,
          "Nearest neighbor 2D upsampling",
          py::arg("x"), py::arg("scale_factor"));
    m.def("relu", &relu,
          "ReLU activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
