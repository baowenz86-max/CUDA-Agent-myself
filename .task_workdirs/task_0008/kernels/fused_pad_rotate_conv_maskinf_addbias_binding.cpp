// Binding for fused_pad_rotate_conv_maskinf_addbias custom op
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the launcher function from .cu file
extern "C" void fused_pad_rotate_conv_maskinf_addbias_launcher(
    const float* input,
    const float* weight,
    const float* conv_bias,
    const float* add_bias,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int K,
    int R,
    int S,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    float pad_value,
    cudaStream_t stream
);

// PyTorch wrapper function
at::Tensor fused_pad_rotate_conv_maskinf_addbias_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor conv_bias,
    at::Tensor add_bias,
    std::tuple<int, int> padding,
    int kernel_size
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(add_bias.is_cuda(), "add_bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(conv_bias.dtype() == torch::kFloat32, "conv_bias must be float32");
    TORCH_CHECK(add_bias.dtype() == torch::kFloat32, "add_bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(add_bias.is_contiguous(), "add_bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K,C,R,S)");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be 1D (K)");
    TORCH_CHECK(add_bias.dim() == 3 && add_bias.size(1) == 1 && add_bias.size(2) == 1, "add_bias must be 3D (K,1,1)");

    // Extract dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = kernel_size;
    int S = kernel_size;  // Assuming square kernel as in original model

    // Padding tuple: (pad_h, pad_w)
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);

    // Check that weight dimensions match input channels and kernel size
    TORCH_CHECK(weight.size(1) == C, "weight input channels must match input tensor");
    TORCH_CHECK(weight.size(2) == R && weight.size(3) == S, "weight kernel size must match provided kernel_size");
    TORCH_CHECK(conv_bias.size(0) == K, "conv_bias size must match output channels");
    TORCH_CHECK(add_bias.size(0) == K, "add_bias size must match output channels");

    // Compute output spatial dimensions
    int outH = H + 2 * pad_h - R + 1;
    int outW = W + 2 * pad_w - S + 1;

    // Allocate output tensor
    auto output = torch::empty({N, K, outH, outW}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pad value: original model used ConstantPad2d with value padding[0], i.e., pad_h (same for all sides).
    // We pass pad_value as pad_h (same as pad_w in original if symmetric).
    float pad_value = static_cast<float>(pad_h);

    // Call launcher
    fused_pad_rotate_conv_maskinf_addbias_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H,
        W,
        K,
        R,
        S,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        pad_value,
        stream
    );

    return output;
}

// Registration function
void register_fused_pad_rotate_conv_maskinf_addbias(pybind11::module& m) {
    m.def("fused_pad_rotate_conv_maskinf_addbias", &fused_pad_rotate_conv_maskinf_addbias_forward,
          "Fused operation: pad + rot90 + conv2d + mask inf fill 0 + add bias",
          py::arg("input"),
          py::arg("weight"),
          py::arg("conv_bias"),
          py::arg("add_bias"),
          py::arg("padding"),
          py::arg("kernel_size"));
}

// Auto-register
REGISTER_BINDING(fused_pad_rotate_conv_maskinf_addbias, register_fused_pad_rotate_conv_maskinf_addbias);
