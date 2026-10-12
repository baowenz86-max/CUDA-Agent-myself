// Binding file for custom CUDA ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void fused_pad_ge_remainder_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int out_H, int out_W,
    long total_elements,
    cudaStream_t stream
);

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int out_H, int out_W,
    long total_elements,
    cudaStream_t stream
);

extern "C" void ge_launcher(
    const float* input,
    bool* output,
    long total_elements,
    cudaStream_t stream
);

extern "C" void remainder_launcher(
    const bool* input,
    int64_t* output,
    long total_elements,
    cudaStream_t stream
);

// Wrapper for fused operation
torch::Tensor custom_ops_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;

    auto output = torch::empty({N, C, out_H, out_W}, input.options().dtype(torch::kInt64));

    long total_elements = static_cast<long>(N) * C * out_H * out_W;
    if (total_elements == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_pad_ge_remainder_launcher(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        out_H, out_W,
        total_elements,
        stream
    );

    return output;
}

// Wrapper for reflection pad only
torch::Tensor reflection_pad2d_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    long total_elements = static_cast<long>(N) * C * out_H * out_W;
    if (total_elements == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        out_H, out_W,
        total_elements,
        stream
    );

    return output;
}

// Wrapper for greater-equal-zero
torch::Tensor ge_forward(torch::Tensor input, double other)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(other == 0.0, "Only comparison with 0 is supported");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    long total_elements = input.numel();
    if (total_elements == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    ge_launcher(
        input.data_ptr<float>(),
        output.data_ptr<bool>(),
        total_elements,
        stream
    );

    return output;
}

// Wrapper for remainder with divisor 2
torch::Tensor remainder_forward(torch::Tensor input, int64_t divisor)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBool, "Input must be bool");
    TORCH_CHECK(divisor == 2, "Only divisor 2 is supported");

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt64));
    long total_elements = input.numel();
    if (total_elements == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    remainder_launcher(
        input.data_ptr<bool>(),
        output.data_ptr<int64_t>(),
        total_elements,
        stream
    );

    return output;
}

// Registration
void register_custom_ops_binding(pybind11::module& m) {
    m.def("custom_ops_forward", &custom_ops_forward,
          "Fused reflection pad + ge + remainder",
          pybind11::arg("input"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));

    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2d",
          pybind11::arg("input"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));

    m.def("ge_forward", &ge_forward,
          "Greater equal zero",
          pybind11::arg("input"),
          pybind11::arg("other") = 0.0);

    m.def("remainder_forward", &remainder_forward,
          "Remainder with divisor",
          pybind11::arg("input"),
          pybind11::arg("divisor") = 2);
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
