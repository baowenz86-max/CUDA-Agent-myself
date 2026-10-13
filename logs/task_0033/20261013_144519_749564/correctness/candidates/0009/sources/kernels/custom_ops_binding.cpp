#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
);

extern "C" void layer_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int channels,
    int length,
    float eps,
    cudaStream_t stream
);

extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int numel,
    cudaStream_t stream
);

// ConvTranspose1d forward wrapper
torch::Tensor conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (in_channels, out_channels, kernel_size)");

    // Ensure contiguity for pointer arithmetic
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    int batch_size = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int in_length = input_contig.size(2);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    TORCH_CHECK(weight_contig.size(0) == in_channels, "Weight in_channels mismatch");
    if (bias.defined()) {
        TORCH_CHECK(bias_contig.numel() == out_channels, "Bias size must match out_channels");
    }

    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias_contig.data_ptr<float>() : nullptr;

    conv_transpose_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}

// LayerNorm forward wrapper
torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int length = input.size(2);

    TORCH_CHECK(weight.numel() == channels * length, "Weight shape mismatch");
    TORCH_CHECK(bias.numel() == channels * length, "Bias shape mismatch");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    layer_norm_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        batch_size,
        channels,
        length,
        eps,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}

// Addcmul forward wrapper
torch::Tensor addcmul_forward(
    torch::Tensor input,
    torch::Tensor tensor1,
    torch::Tensor tensor2
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(tensor1.is_cuda(), "Tensor1 must be a CUDA tensor");
    TORCH_CHECK(tensor2.is_cuda(), "Tensor2 must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(tensor1.dtype() == torch::kFloat32, "Tensor1 must be float32");
    TORCH_CHECK(tensor2.dtype() == torch::kFloat32, "Tensor2 must be float32");
    TORCH_CHECK(input.sizes() == tensor1.sizes() && input.sizes() == tensor2.sizes(), "All tensors must have the same shape");

    auto input_contig = input.contiguous();
    auto tensor1_contig = tensor1.contiguous();
    auto tensor2_contig = tensor2.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    addcmul_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        tensor1_contig.data_ptr<float>(),
        tensor2_contig.data_ptr<float>(),
        input.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose_forward", &conv_transpose_forward,
          "ConvTranspose1d forward (custom CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
    m.def("addcmul_forward", &addcmul_forward,
          "Addcmul forward (value=1.0)",
          py::arg("input"), py::arg("tensor1"), py::arg("tensor2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
