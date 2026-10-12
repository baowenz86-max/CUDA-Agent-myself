#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv2d_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_h, int kernel_w,
    cudaStream_t stream);

extern "C" void triu_forward_launcher(
    const float* input, float* output,
    int batch_size, int channels, int height, int width,
    cudaStream_t stream);

extern "C" void bitwise_xor_forward_launcher(
    const uint8_t* input, const uint8_t* bias, uint8_t* output,
    int batch_size, int channels, int height, int width,
    cudaStream_t stream);

extern "C" void cast_float_forward_launcher(
    const uint8_t* input, float* output,
    int total,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// Conv2d wrapper
// -----------------------------------------------------------------------------
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels, kh, kw)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    TORCH_CHECK(weight.size(1) == in_channels, "weight in_channels must match input channels");
    TORCH_CHECK(bias.numel() == out_channels, "bias must have one element per output channel");

    // Assume stride=1, padding=0 (as in reference model)
    int out_height = in_height - kernel_h + 1;
    int out_width = in_width - kernel_w + 1;
    TORCH_CHECK(out_height > 0 && out_width > 0, "invalid output dimensions");

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_h, kernel_w,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Triu wrapper
// -----------------------------------------------------------------------------
torch::Tensor triu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Bitwise XOR wrapper
// -----------------------------------------------------------------------------
torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(bias.dtype() == torch::kUInt8, "bias must be uint8");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Bias must have one element per channel; may be shape [C,1,1] or [C]
    TORCH_CHECK(bias.numel() == channels, "bias must have one element per channel");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bitwise_xor_forward_launcher(
        input.data_ptr<uint8_t>(),
        bias.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        batch_size, channels, height, width,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Cast uint8 to float wrapper
// -----------------------------------------------------------------------------
torch::Tensor cast_float_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cast_float_forward_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward (stride=1, no padding)",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("triu_forward", &triu_forward, "Upper triangular mask on each channel",
          py::arg("input"));
    m.def("bitwise_xor_forward", &bitwise_xor_forward, "Bitwise XOR with per-channel bias",
          py::arg("input"), py::arg("bias"));
    m.def("cast_float_forward", &cast_float_forward, "Cast uint8 to float32",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
