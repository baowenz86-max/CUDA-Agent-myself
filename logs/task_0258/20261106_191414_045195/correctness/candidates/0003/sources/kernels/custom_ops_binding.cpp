#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <iostream>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int KH, int KW,
    int OH, int OW,
    int stride_h, int stride_w,
    cudaStream_t stream
);

extern "C" void relu_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
);

extern "C" void maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int OH, int OW,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
);

// Helper to check CUDA errors
static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

// conv2d_forward
// Expected input: [N, C, H, W], weight: [K, C, KH, KW], bias: [K] or undefined
// stride_h, stride_w, padding_h, padding_w are int64_t but we only support stride=1, padding=0 per contract
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(stride_h == 1 && stride_w == 1, "only stride=1 supported");
    TORCH_CHECK(padding_h == 0 && padding_w == 0, "only padding=0 supported");

    // Ensure contiguous for simplicity; we handle strides by making contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);
    int K = weight_contig.size(0);
    int KH = weight_contig.size(2);
    int KW = weight_contig.size(3);
    TORCH_CHECK(weight_contig.size(1) == C, "input channels must match weight in_channels");

    int OH = (H + 2 * padding_h - KH) / stride_h + 1;
    int OW = (W + 2 * padding_w - KW) / stride_w + 1;
    TORCH_CHECK(OH > 0 && OW > 0, "invalid output size");

    auto output = torch::empty({N, K, OH, OW}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, K, KH, KW, OH, OW,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        stream
    );
    check_cuda_error();

    return output;
}

// relu_forward
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int total = input_contig.numel();
    relu_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), total, stream);
    check_cuda_error();

    return output;
}

// max_pool2d_forward
torch::Tensor max_pool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(dilation == 1, "only dilation=1 supported");

    auto input_contig = input.contiguous();
    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(OH > 0 && OW > 0, "invalid output size");

    auto output = torch::empty({N, C, OH, OW}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool2d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, OH, OW,
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
        stream
    );
    check_cuda_error();

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward (custom CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride_h"), py::arg("stride_w"), py::arg("padding_h"), py::arg("padding_w"));
    m.def("relu_forward", &relu_forward, "ReLU forward (custom CUDA)", py::arg("input"));
    m.def("max_pool2d_forward", &max_pool2d_forward, "MaxPool2d forward (custom CUDA)",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
