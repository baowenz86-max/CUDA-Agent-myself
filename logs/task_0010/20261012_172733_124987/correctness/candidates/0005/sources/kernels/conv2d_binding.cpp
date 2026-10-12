#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    cudaStream_t stream);

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (K, C, R, S)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D (K)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);

    TORCH_CHECK(weight.size(1) == C, "Weight input channels must match input channels");
    TORCH_CHECK(bias.size(0) == K, "Bias size must match output channels");
    TORCH_CHECK(H >= R && W >= S, "Input spatial dimensions must be >= kernel size");

    int H_out = H - R + 1;
    int W_out = W - S + 1;
    auto output = torch::empty({N, K, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        K, R, S,
        stream
    );

    return output;
}

void register_conv2d_forward(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward");
}

REGISTER_BINDING(conv2d, register_conv2d_forward);
