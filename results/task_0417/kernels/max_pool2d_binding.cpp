#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void max_pool2d_with_indices_forward_launcher(
    const float* input,
    float* pooled,
    int64_t* indices,
    int N, int C, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> max_pool2d_with_indices_forward(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel sizes must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "strides must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");

    int H_out = (H_in + 2 * padding_h - (kernel_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - (kernel_w - 1) - 1) / stride_w + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "output size must be positive");

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(input.device());

    torch::Tensor pooled = torch::empty({N, C, H_out, W_out}, options_float);
    torch::Tensor indices = torch::empty({N, C, H_out, W_out}, options_int64);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_pool2d_with_indices_forward_launcher(
        input.data_ptr<float>(),
        pooled.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H_in, W_in,
        static_cast<int>(kernel_h), static_cast<int>(kernel_w),
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(padding_h), static_cast<int>(padding_w),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(pooled, indices);
}

void register_max_pool2d_with_indices(pybind11::module& m) {
    m.def("max_pool2d_with_indices_forward", &max_pool2d_with_indices_forward,
          "2D max pooling with indices (PyTorch semantics)",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));
}

REGISTER_BINDING(max_pool2d_with_indices, register_max_pool2d_with_indices);
