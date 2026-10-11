#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include "../binding_registry.h"

// Declaration of the launcher from bitwise_xor_uint8_forward.cu
extern "C" void bitwise_xor_uint8_forward_launcher(
    uint8_t* output,
    const uint8_t* x,
    const uint8_t* bias,
    int64_t num_elements,
    int64_t HW,
    int64_t C,
    cudaStream_t stream);

// PyTorch wrapper
at::Tensor bitwise_xor_uint8_forward(at::Tensor x, at::Tensor bias) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kByte, "x must be uint8");
    TORCH_CHECK(bias.dtype() == at::kByte, "bias must be uint8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // x shape: (N, C, H, W)
    // bias shape: (C, 1, 1)
    auto x_sizes = x.sizes();
    auto bias_sizes = bias.sizes();
    TORCH_CHECK(x_sizes.size() == 4, "x must be 4D");
    TORCH_CHECK(bias_sizes.size() == 3, "bias must be 3D");
    TORCH_CHECK(bias_sizes[0] == x_sizes[1], "bias size[0] must match x's channel dimension");
    TORCH_CHECK(bias_sizes[1] == 1 && bias_sizes[2] == 1, "bias must have shape (C, 1, 1)");

    int64_t N = x_sizes[0];
    int64_t C = x_sizes[1];
    int64_t H = x_sizes[2];
    int64_t W = x_sizes[3];
    int64_t HW = H * W;
    int64_t num_elements = x.numel();

    // Allocate output tensor with same shape as x
    auto output = at::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    bitwise_xor_uint8_forward_launcher(
        output.data_ptr<uint8_t>(),
        x.data_ptr<uint8_t>(),
        bias.data_ptr<uint8_t>(),
        num_elements,
        HW,
        C,
        stream);

    return output;
}

// Registration function
void register_bitwise_xor_uint8_forward(pybind11::module& m) {
    m.def("bitwise_xor_uint8_forward", &bitwise_xor_uint8_forward,
          "Bitwise XOR with broadcasting for uint8 tensors",
          pybind11::arg("x"), pybind11::arg("bias"));
}

// Auto-register
REGISTER_BINDING(bitwise_xor_uint8_forward, register_bitwise_xor_uint8_forward);
