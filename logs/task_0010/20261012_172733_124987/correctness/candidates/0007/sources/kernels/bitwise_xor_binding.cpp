#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher from .cu file
extern "C" void bitwise_xor_launcher(
    unsigned char* out,
    const unsigned char* in,
    const float* bias,
    int C,
    int spatial_size,
    int N,
    cudaStream_t stream);

// Wrapper function: accepts x (any float dtype or uint8) and bias (any shape)
torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dim() >= 3, "Input must have at least 3 dimensions (N, C, ...)");

    // Determine channel dimension C (second dimension)
    int C = input.size(1);
    // Bias must have exactly C elements (any shape)
    TORCH_CHECK(bias.numel() == C, "Bias must have exactly ", C, " elements (got ", bias.numel(), ")");

    // Cast input to uint8 if not already
    torch::Tensor input_u8;
    if (input.dtype() != torch::kUInt8) {
        input_u8 = input.to(torch::kUInt8);
    } else {
        input_u8 = input;
    }

    // Cast bias to float32 (the kernel expects float pointer) if needed
    torch::Tensor bias_f32;
    if (bias.dtype() != torch::kFloat32) {
        bias_f32 = bias.to(torch::kFloat32);
    } else {
        bias_f32 = bias;
    }

    // Ensure bias is contiguous after casting
    if (!bias_f32.is_contiguous()) {
        bias_f32 = bias_f32.contiguous();
    }

    // Output tensor: same shape as input, dtype uint8
    auto output = torch::empty_like(input_u8);
    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    int N = input_u8.numel();
    int spatial_size = N / C; // assuming N = batch * C * H * W, so spatial_size = batch*H*W
    // The kernel expects C = number of channels, spatial_size = number of elements per channel including batch? Actually it uses c = (i / spatial_size) % C.
    // For NCHW layout: index i = ((n*C + c)*H + h)*W + w. If we set spatial_size = H*W (per channel per batch?), but kernel uses c = (i / spatial_size) % C. That only works if spatial_size is H*W and N = batch*C*H*W. Then i / (H*W) gives linear index of (n*C + c), mod C gives c. So spatial_size should be H*W, not N/C. In our case, input shape is (B, C, H, W). So H*W = input.size(2)*input.size(3). We'll compute that.
    int H = input_u8.size(2);
    int W = input_u8.size(3);
    int spatial_size = H * W;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bitwise_xor_launcher(
        output.data_ptr<unsigned char>(),
        input_u8.data_ptr<unsigned char>(),
        bias_f32.data_ptr<float>(),
        C,
        spatial_size,
        N,
        stream
    );

    return output;
}

void register_bitwise_xor_forward(pybind11::module& m) {
    m.def("bitwise_xor_forward", &bitwise_xor_forward,
          "Bitwise XOR forward (uint8)",
          pybind11::arg("input"), pybind11::arg("bias"));
}

REGISTER_BINDING(bitwise_xor_forward, register_bitwise_xor_forward);