#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void logsoftmax_forward_launcher(const float* input,
                                            float* output,
                                            int N, int C, int D, int H, int W,
                                            cudaStream_t stream);

extern "C" void maxpool3d_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int D, int H, int W,
                                           int kernel_d, int kernel_h, int kernel_w,
                                           int stride_d, int stride_h, int stride_w,
                                           int padding_d, int padding_h, int padding_w,
                                           cudaStream_t stream);

static torch::Tensor logsoftmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    auto output = torch::empty_like(input);

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    logsoftmax_forward_launcher(input.data_ptr<float>(),
                                output.data_ptr<float>(),
                                N, C, D, H, W,
                                stream);
    return output;
}

static torch::Tensor maxpool3d_forward(torch::Tensor input,
                                       int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
                                       int64_t stride_d, int64_t stride_h, int64_t stride_w,
                                       int64_t padding_d, int64_t padding_h, int64_t padding_w) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N,C,D,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int D_out = (D + 2 * static_cast<int>(padding_d) - static_cast<int>(kernel_d)) / static_cast<int>(stride_d) + 1;
    int H_out = (H + 2 * static_cast<int>(padding_h) - static_cast<int>(kernel_h)) / static_cast<int>(stride_h) + 1;
    int W_out = (W + 2 * static_cast<int>(padding_w) - static_cast<int>(kernel_w)) / static_cast<int>(stride_w) + 1;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool3d_forward_launcher(input.data_ptr<float>(),
                               output.data_ptr<float>(),
                               N, C, D, H, W,
                               static_cast<int>(kernel_d), static_cast<int>(kernel_h), static_cast<int>(kernel_w),
                               static_cast<int>(stride_d), static_cast<int>(stride_h), static_cast<int>(stride_w),
                               static_cast<int>(padding_d), static_cast<int>(padding_h), static_cast<int>(padding_w),
                               stream);
    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("logsoftmax_forward", &logsoftmax_forward,
          "LogSoftmax over channel dimension (dim=1)",
          py::arg("input"), py::arg("dim"));
    m.def("maxpool3d_forward", &maxpool3d_forward,
          "3D Max Pooling forward",
          py::arg("input"),
          py::arg("kernel_d"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_d"), py::arg("stride_h"), py::arg("stride_w"),
          py::arg("padding_d"), py::arg("padding_h"), py::arg("padding_w"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
