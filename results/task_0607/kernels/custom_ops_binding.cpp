#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void launch_custom_unfold_abs(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    cudaStream_t stream);

torch::Tensor custom_unfold_abs(
    torch::Tensor x,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride_h,
    int64_t stride_w) {

    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional [N, C, H, W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // 计算输出空间维度 L
    int64_t out_h = (H + 2 * padding_h - dilation * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * padding_w - dilation * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "Invalid output dimensions");
    int64_t out_spatial = out_h * out_w;
    int64_t CK = C * kernel_h * kernel_w;

    auto output = torch::empty({N, CK, out_spatial}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_custom_unfold_abs(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(dilation),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        stream);

    // 检查 kernel 启动错误（可选）
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_unfold_abs(pybind11::module& m) {
    m.def("custom_unfold_abs", &custom_unfold_abs,
          "Custom unfold + abs operation",
          py::arg("x"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation"),
          py::arg("padding_h"),
          py::arg("padding_w"),
          py::arg("stride_h"),
          py::arg("stride_w"));
}

REGISTER_BINDING(custom_unfold_abs, register_custom_unfold_abs);
