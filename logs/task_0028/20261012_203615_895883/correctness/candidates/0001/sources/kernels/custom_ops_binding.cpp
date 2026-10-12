#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long sB, long long sC, long long sH, long long sW,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    cudaStream_t stream
);

extern "C" void sign_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream
);

extern "C" void cos_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream
);

// unfold_forward: input (B, C, H, W) -> output (B, C*kh*kw, L)
torch::Tensor unfold_forward(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride_h,
    int64_t stride_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (B, C, H, W)");

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int out_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int out_channels = C * kernel_h * kernel_w;
    long long L = (long long)out_h * out_w;

    TORCH_CHECK(out_h > 0 && out_w > 0, "Invalid output dimensions");

    auto output = torch::empty({B, out_channels, L}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    unfold_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        padding_h, padding_w,
        stride_h, stride_w,
        stream
    );

    return output;
}

// sign_forward: elementwise sign
torch::Tensor sign_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sign_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// cos_forward: elementwise cosine
torch::Tensor cos_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cos_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("unfold_forward", &unfold_forward, "Unfold forward",
          py::arg("input"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("dilation_h"), py::arg("dilation_w"),
          py::arg("padding_h"), py::arg("padding_w"),
          py::arg("stride_h"), py::arg("stride_w"));
    m.def("sign_forward", &sign_forward, "Sign forward",
          py::arg("input"));
    m.def("cos_forward", &cos_forward, "Cos forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
