#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int pad_d, int pad_h, int pad_w,
    cudaStream_t stream);

extern "C" void avg_pool3d_double_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int k1_d, int k1_h, int k1_w,
    int k2_d, int k2_h, int k2_w,
    int stride, int pad_d, int pad_h, int pad_w,
    cudaStream_t stream);

// 辅助函数：检查输入参数并返回连续 CUDA float32 张量
static torch::Tensor check_avg_pool_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "avg_pool3d input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "avg_pool3d input must be float32");
    TORCH_CHECK(input.dim() == 5, "avg_pool3d input must be 5D");
    if (!input.is_contiguous()) {
        return input.contiguous();
    }
    return input;
}

torch::Tensor avg_pool3d_forward(
    const torch::Tensor& input,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride,
    int64_t pad_d, int64_t pad_h, int64_t pad_w) {

    auto x = check_avg_pool_input(input);
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    int outD = (D + 2 * static_cast<int>(pad_d) - static_cast<int>(kernel_d)) / static_cast<int>(stride) + 1;
    int outH = (H + 2 * static_cast<int>(pad_h) - static_cast<int>(kernel_h)) / static_cast<int>(stride) + 1;
    int outW = (W + 2 * static_cast<int>(pad_w) - static_cast<int>(kernel_w)) / static_cast<int>(stride) + 1;

    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "avg_pool3d: invalid output dimensions");

    auto output = torch::empty({N, C, outD, outH, outW}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        static_cast<int>(kernel_d), static_cast<int>(kernel_h), static_cast<int>(kernel_w),
        static_cast<int>(stride),
        static_cast<int>(pad_d), static_cast<int>(pad_h), static_cast<int>(pad_w),
        stream);

    return output;
}

torch::Tensor avg_pool3d_double_forward(
    const torch::Tensor& input,
    int64_t kernel1_d, int64_t kernel1_h, int64_t kernel1_w,
    int64_t kernel2_d, int64_t kernel2_h, int64_t kernel2_w,
    int64_t stride,
    int64_t pad_d, int64_t pad_h, int64_t pad_w) {

    auto x = check_avg_pool_input(input);
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    int int_stride = static_cast<int>(stride);
    int int_pad_d = static_cast<int>(pad_d);
    int int_pad_h = static_cast<int>(pad_h);
    int int_pad_w = static_cast<int>(pad_w);

    int out1D = (D + 2 * int_pad_d - static_cast<int>(kernel1_d)) / int_stride + 1;
    int out1H = (H + 2 * int_pad_h - static_cast<int>(kernel1_h)) / int_stride + 1;
    int out1W = (W + 2 * int_pad_w - static_cast<int>(kernel1_w)) / int_stride + 1;

    int out2D = (out1D + 2 * int_pad_d - static_cast<int>(kernel2_d)) / int_stride + 1;
    int out2H = (out1H + 2 * int_pad_h - static_cast<int>(kernel2_h)) / int_stride + 1;
    int out2W = (out1W + 2 * int_pad_w - static_cast<int>(kernel2_w)) / int_stride + 1;

    TORCH_CHECK(out2D > 0 && out2H > 0 && out2W > 0, "avg_pool3d_double: invalid output dimensions");

    auto output = torch::empty({N, C, out2D, out2H, out2W}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool3d_double_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        static_cast<int>(kernel1_d), static_cast<int>(kernel1_h), static_cast<int>(kernel1_w),
        static_cast<int>(kernel2_d), static_cast<int>(kernel2_h), static_cast<int>(kernel2_w),
        int_stride,
        int_pad_d, int_pad_h, int_pad_w,
        stream);

    return output;
}

void register_avg_pool3d(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "3D average pooling forward",
          py::arg("input"),
          py::arg("kernel_d"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride"),
          py::arg("pad_d"), py::arg("pad_h"), py::arg("pad_w"));

    m.def("avg_pool3d_double_forward", &avg_pool3d_double_forward,
          "Fused double 3D average pooling forward",
          py::arg("input"),
          py::arg("kernel1_d"), py::arg("kernel1_h"), py::arg("kernel1_w"),
          py::arg("kernel2_d"), py::arg("kernel2_h"), py::arg("kernel2_w"),
          py::arg("stride"),
          py::arg("pad_d"), py::arg("pad_h"), py::arg("pad_w"));
}

REGISTER_BINDING(avg_pool3d, register_avg_pool3d);
