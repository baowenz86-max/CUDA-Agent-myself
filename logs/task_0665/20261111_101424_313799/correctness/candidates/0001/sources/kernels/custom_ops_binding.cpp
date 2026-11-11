#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void channel_mean_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream);

// channel_mean_forward wrapper
torch::Tensor channel_mean_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    // Allocate output [N, 1, H, W] with same options as input
    auto output = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    channel_mean_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    return output;
}

// max_pool2d_forward wrapper
torch::Tensor max_pool2d_forward(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int64_t outH = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t outW = (W + 2 * padding_w - kernel_w) / stride_w + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "invalid pooling parameters");

    auto output = torch::empty({N, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops_binding(pybind11::module& m) {
    m.def("channel_mean_forward", &channel_mean_forward, "Channel mean forward",
          py::arg("x"));
    m.def("max_pool2d_forward", &max_pool2d_forward, "MaxPool2d forward",
          py::arg("x"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
