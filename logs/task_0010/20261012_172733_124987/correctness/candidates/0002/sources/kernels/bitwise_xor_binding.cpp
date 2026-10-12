#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher defined in bitwise_xor_forward.cu
extern "C" void bitwise_xor_launcher(
    unsigned char* output,
    const unsigned char* input,
    const unsigned char* bias,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream
);

torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(bias.dtype() == torch::kUInt8, "Bias must be uint8");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1, "Bias must be shape (C,1,1)");

    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    TORCH_CHECK(bias.size(0) == channels, "Bias channels mismatch");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bitwise_xor_launcher(
        output.data_ptr<unsigned char>(),
        input.data_ptr<unsigned char>(),
        bias.data_ptr<unsigned char>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    return output;
}

void register_bitwise_xor_forward(pybind11::module& m) {
    m.def("bitwise_xor_forward", &bitwise_xor_forward, "Bitwise XOR with broadcast bias");
}

REGISTER_BINDING(bitwise_xor_forward, register_bitwise_xor_forward);
