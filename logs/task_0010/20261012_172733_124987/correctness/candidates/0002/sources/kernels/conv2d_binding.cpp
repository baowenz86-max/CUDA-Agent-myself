#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher defined in conv2d_forward.cu
extern "C" void conv2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream
);

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC,IC,KH,KW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels, "weight channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias size mismatch");
    TORCH_CHECK(kernel_h % 2 == 1 && kernel_w % 2 == 1, "kernel must be odd");
    int out_h = (in_h + 2 * (kernel_h / 2) - kernel_h) + 1;
    int out_w = (in_w + 2 * (kernel_w / 2) - kernel_w) + 1;

    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stream
    );

    return output;
}

void register_conv2d_forward(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward using cuDNN");
}

REGISTER_BINDING(conv2d_forward, register_conv2d_forward);
