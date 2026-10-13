#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void conv_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void upsample_nearest_forward(
    const float* input,
    float* output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    cudaStream_t stream);

extern "C" void one_hot_forward(
    const int64_t* input,
    int64_t* output,
    int total_elements,
    int num_classes,
    cudaStream_t stream);

// ConvTranspose binding
torch::Tensor conv_transpose_forward_binding(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,  // nullable
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in, C_out, kH, kW)");
    if (bias.defined()) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    }

    int batch = input.size(0);
    int in_channels = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight C_in must match input channels");
    TORCH_CHECK(kernel_size == weight.size(3), "kernel height and width must be equal");
    if (bias.defined()) {
        TORCH_CHECK(bias.size(0) == out_channels, "bias size must match out_channels");
    }

    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch, out_channels, H_out, W_out}, input.options());

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        H_in,
        W_in,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after conv_transpose_forward: ", cudaGetErrorString(err));

    return output;
}

// Upsample nearest binding
torch::Tensor upsample_nearest_forward_binding(
    torch::Tensor input,
    int64_t scale_factor)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");

    int batch = input.size(0);
    int channels = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int H_out = H_in * scale_factor;
    int W_out = W_in * scale_factor;

    auto output = torch::empty({batch, channels, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    upsample_nearest_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        H_in,
        W_in,
        static_cast<int>(scale_factor),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after upsample_nearest_forward: ", cudaGetErrorString(err));

    return output;
}

// One-hot binding
torch::Tensor one_hot_forward_binding(
    torch::Tensor input,
    int64_t num_classes)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kLong, "input must be int64");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    auto input_sizes = input.sizes().vec();
    input_sizes.push_back(num_classes);
    auto output = torch::empty(input_sizes, input.options());

    int total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    one_hot_forward(
        input.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(),
        total_elements,
        static_cast<int>(num_classes),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after one_hot_forward: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose_forward", &conv_transpose_forward_binding,
          "ConvTranspose2d forward (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("upsample_nearest_forward", &upsample_nearest_forward_binding,
          "Nearest neighbor upsample forward",
          pybind11::arg("input"),
          pybind11::arg("scale_factor"));
    m.def("one_hot_forward", &one_hot_forward_binding,
          "One-hot encoding forward",
          pybind11::arg("input"),
          pybind11::arg("num_classes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
