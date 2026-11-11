#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void custom_masked_fill_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    float mask_value,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream
);

void custom_cat_launcher(
    float* output,
    const float* x,
    const float* y,
    int64_t B, int64_t C1, int64_t C2, int64_t H, int64_t W,
    int64_t x_stride_b, int64_t x_stride_c, int64_t x_stride_h, int64_t x_stride_w,
    int64_t y_stride_b, int64_t y_stride_c, int64_t y_stride_h, int64_t y_stride_w,
    cudaStream_t stream
);
}

// Wrapper for custom_masked_fill
// Input: x (float32 CUDA tensor), mask_value (double)
// Output: tensor with same shape and strides as x, filled with 0 where x < mask_value
static torch::Tensor custom_masked_fill(torch::Tensor x, double mask_value) {
    TORCH_CHECK(x.is_cuda(), "custom_masked_fill: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_masked_fill: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_masked_fill: input must be 4D (B, C, H, W)");

    auto output = torch::empty_like(x);

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t stride_b = x.stride(0);
    int64_t stride_c = x.stride(1);
    int64_t stride_h = x.stride(2);
    int64_t stride_w = x.stride(3);
    int64_t total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_masked_fill_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        total_elements,
        static_cast<float>(mask_value),
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_masked_fill: kernel launch failed");
    return output;
}

// Wrapper for custom_cat
// Inputs: x, y (float32 CUDA tensors), dim (int64, only dim=1 supported)
// Output: new contiguous tensor with concatenation along dim
static torch::Tensor custom_cat(torch::Tensor x, torch::Tensor y, int64_t dim) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "custom_cat: inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32, "custom_cat: inputs must be float32");
    TORCH_CHECK(x.dim() == 4 && y.dim() == 4, "custom_cat: inputs must be 4D");
    TORCH_CHECK(dim == 1, "custom_cat: only dim=1 is supported");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(2) == y.size(2) && x.size(3) == y.size(3),
                "custom_cat: input shapes must match except in channel dimension");

    int64_t B = x.size(0);
    int64_t C1 = x.size(1);
    int64_t C2 = y.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output as contiguous tensor with shape [B, C1+C2, H, W]
    auto output = torch::empty({B, C1 + C2, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_cat_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, C1, C2, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_cat: kernel launch failed");
    return output;
}

// Register bindings
static void register_custom_ops(pybind11::module& m) {
    m.def("custom_masked_fill", &custom_masked_fill,
          "Custom masked_fill (x < mask_value -> 0)",
          py::arg("x"), py::arg("mask_value") = 0.5);
    m.def("custom_cat", &custom_cat,
          "Custom concatenate along dim=1",
          py::arg("x"), py::arg("y"), py::arg("dim") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
