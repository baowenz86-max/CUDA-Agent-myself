#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <cstdint>

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_uint8_launcher(
    const uint8_t* input,
    uint8_t* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int padding,
    cudaStream_t stream);

extern "C" void bitwise_xor_uint8_launcher(
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* output,
    int64_t total,
    cudaStream_t stream);

// ReflectionPad2d wrapper
torch::Tensor reflection_pad2d_uint8(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "reflection_pad2d_uint8: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "reflection_pad2d_uint8: input must be uint8");
    TORCH_CHECK(input.is_contiguous(), "reflection_pad2d_uint8: input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "reflection_pad2d_uint8: input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding >= 0, "reflection_pad2d_uint8: padding must be non-negative");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto H_out = H_in + 2 * padding;
    auto W_out = W_in + 2 * padding;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_uint8_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        N, C, H_in, W_in, H_out, W_out,
        static_cast<int>(padding),
        stream);

    return output;
}

// Bitwise XOR wrapper
torch::Tensor bitwise_xor_uint8(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "bitwise_xor_uint8: tensors must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kUInt8 && b.dtype() == torch::kUInt8,
                "bitwise_xor_uint8: tensors must be uint8");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "bitwise_xor_uint8: tensors must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "bitwise_xor_uint8: tensors must have identical shape");

    auto output = torch::empty_like(a);
    int64_t total = a.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bitwise_xor_uint8_launcher(
        a.data_ptr<uint8_t>(),
        b.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        total,
        stream);

    return output;
}

// Register both operators
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_uint8", &reflection_pad2d_uint8,
          "ReflectionPad2d for uint8 tensors",
          py::arg("input"),
          py::arg("padding"));
    m.def("bitwise_xor_uint8", &bitwise_xor_uint8,
          "Elementwise bitwise XOR for uint8 tensors",
          py::arg("a"),
          py::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
