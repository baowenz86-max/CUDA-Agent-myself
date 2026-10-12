#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" {
    void conv2d_forward_launcher(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int N, int C, int H, int W,
        int K, int R, int S,
        int outH, int outW,
        cudaStream_t stream);

    void triu_forward_launcher(
        const float* input,
        float* output,
        int N, int C, int H, int W,
        cudaStream_t stream);

    void bitwise_xor_forward_launcher(
        const uint8_t* input,
        const uint8_t* other,
        uint8_t* output,
        int total,
        cudaStream_t stream);

    void cast_float_forward_launcher(
        const uint8_t* input,
        float* output,
        int total,
        cudaStream_t stream);
}

// ------------------------- Conv2d Forward -------------------------
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K,C,R,S)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (K)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);
    TORCH_CHECK(weight.size(1) == C, "weight channel dimension must match input channels");
    TORCH_CHECK(bias.size(0) == K, "bias size must equal number of output channels");

    int outH = H - R + 1;
    int outW = W - S + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "output dimensions must be positive");

    auto output = torch::empty({N, K, outH, outW}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, K, R, S, outH, outW,
        stream);
    return output;
}

// ------------------------- Triu Forward -------------------------
torch::Tensor triu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D for triu on last two dims");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        stream);
    return output;
}

// ------------------------- Bitwise XOR Forward -------------------------
torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor other) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(other.is_cuda(), "other must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(other.is_contiguous(), "other must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(other.dtype() == torch::kUInt8, "other must be uint8");
    TORCH_CHECK(input.numel() == other.numel(), "tensors must have the same number of elements");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bitwise_xor_forward_launcher(
        input.data_ptr<uint8_t>(),
        other.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        static_cast<int>(input.numel()),
        stream);
    return output;
}

// ------------------------- Cast Float Forward -------------------------
torch::Tensor cast_float_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cast_float_forward_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream);
    return output;
}

// ------------------------- Registration -------------------------
void register_custom_ops(py::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "2D convolution with bias (stride=1, padding=0)");
    m.def("triu_forward", &triu_forward, "Upper triangular part on last two dims");
    m.def("bitwise_xor_forward", &bitwise_xor_forward, "Bitwise XOR of two uint8 tensors");
    m.def("cast_float_forward", &cast_float_forward, "Cast uint8 tensor to float32");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
