#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void bitwise_xor_launcher(
    const uint8_t* input1,
    const uint8_t* bias,
    uint8_t* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(bias.dtype() == torch::kUInt8, "Bias must be uint8");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(bias.numel() == C, "Bias must have exactly C elements matching input channel dimension");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bitwise_xor_launcher(
        input.data_ptr<uint8_t>(),
        bias.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        N, C, H, W,
        stream
    );

    return output;
}

void register_bitwise_xor_forward(pybind11::module& m) {
    m.def("bitwise_xor_forward", &bitwise_xor_forward, "Bitwise XOR forward");
}

REGISTER_BINDING(bitwise_xor, register_bitwise_xor_forward);