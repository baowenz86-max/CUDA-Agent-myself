// Binding for custom_max_dim and custom_max_pool2d CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <cstdint>
#include <vector>

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int B, int C, int H, int W,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream);

extern "C" void custom_max_pool2d_launcher(
    const float* input,
    float* output,
    int B, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation,
    long long input_stride_b,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream);

// ---------------------------------------------------------------------
// custom_max_dim forward
// ---------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> custom_max_dim_forward(
    torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "custom_max_dim: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_dim: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_dim: x must be 4D [B,C,H,W]");
    TORCH_CHECK(dim == 1, "custom_max_dim: only dim=1 is supported");
    TORCH_CHECK(keepdim, "custom_max_dim: keepdim must be true");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "custom_max_dim: input dimensions must be positive");

    // Output shape [B,1,H,W]
    auto values = torch::empty({B, 1, H, W}, x.options());
    auto indices = torch::empty({B, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides of original input to support non-contiguous tensors.
    custom_max_dim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        static_cast<int>(B), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<long long>(x.stride(0)),
        static_cast<long long>(x.stride(1)),
        static_cast<long long>(x.stride(2)),
        static_cast<long long>(x.stride(3)),
        static_cast<long long>(values.stride(0)),
        static_cast<long long>(values.stride(2)),
        static_cast<long long>(values.stride(3)),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_dim: CUDA launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

// ---------------------------------------------------------------------
// custom_max_pool2d forward
// ---------------------------------------------------------------------
torch::Tensor custom_max_pool2d_forward(
    torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t dilation, bool ceil_mode) {
    TORCH_CHECK(x.is_cuda(), "custom_max_pool2d: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_pool2d: x must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_pool2d: x must be 4D [B,C,H,W]");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0,
                "custom_max_pool2d: kernel_size, stride, dilation must be positive; padding non-negative");
    TORCH_CHECK(!ceil_mode, "custom_max_pool2d: only ceil_mode=false is supported");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C == 1, "custom_max_pool2d: input must have C=1");
    TORCH_CHECK(B > 0 && H > 0 && W > 0, "custom_max_pool2d: input dimensions must be positive");

    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "custom_max_pool2d: output dimensions must be positive");

    auto output = torch::empty({B, 1, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides of input to support non-contiguous tensors.
    custom_max_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(H_out), static_cast<int>(W_out),
        static_cast<int>(kernel_size), static_cast<int>(stride),
        static_cast<int>(padding), static_cast<int>(dilation),
        static_cast<long long>(x.stride(0)),
        static_cast<long long>(x.stride(2)),
        static_cast<long long>(x.stride(3)),
        static_cast<long long>(output.stride(0)),
        static_cast<long long>(output.stride(2)),
        static_cast<long long>(output.stride(3)),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_pool2d: CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim", &custom_max_dim_forward,
          "Max over dimension 1 with keepdim=True",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("custom_max_pool2d", &custom_max_pool2d_forward,
          "2D max pooling with specified kernel/stride/padding/dilation",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
