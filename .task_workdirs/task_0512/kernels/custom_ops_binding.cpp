#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the launcher from custom_ops.cu
extern "C" void avgpool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_pixel_stride,
    long long output_batch_stride,
    long long output_channel_stride,
    long long output_pixel_stride,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
);

// PyTorch binding function
static torch::Tensor avgpool1d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must have 3 dimensions (batch, channels, length)");

    // Validate kernel parameters
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Cast to int for CUDA launcher, check range
    TORCH_CHECK(kernel_size <= INT32_MAX, "kernel_size too large");
    TORCH_CHECK(stride <= INT32_MAX, "stride too large");
    TORCH_CHECK(padding <= INT32_MAX, "padding too large");
    int kernel_size_i = static_cast<int>(kernel_size);
    int stride_i = static_cast<int>(stride);
    int padding_i = static_cast<int>(padding);

    // Extract dimensions
    int batch_size = static_cast<int>(input.size(0));
    int channels = static_cast<int>(input.size(1));
    int input_length = static_cast<int>(input.size(2));

    // Compute output length: floor((input_length + 2*padding - kernel_size) / stride) + 1
    int output_length = (input_length + 2 * padding_i - kernel_size_i) / stride_i + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    // Allocate output tensor with same shape but last dim = output_length
    // Use torch::empty_like(input) then view/slice? Better: allocate new tensor with required shape.
    // We need output shape [batch, channels, output_length], same device and dtype.
    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    // Get strides. Input might be non-contiguous; we pass actual strides to launcher.
    long long input_batch_stride = input.stride(0);
    long long input_channel_stride = input.stride(1);
    long long input_pixel_stride = input.stride(2);
    long long output_batch_stride = output.stride(0);
    long long output_channel_stride = output.stride(1);
    long long output_pixel_stride = output.stride(2);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avgpool1d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        input_batch_stride,
        input_channel_stride,
        input_pixel_stride,
        output_batch_stride,
        output_channel_stride,
        output_pixel_stride,
        kernel_size_i,
        stride_i,
        padding_i,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_avgpool1d(pybind11::module& m) {
    m.def("avgpool1d_forward", &avgpool1d_forward,
          "Average pooling 1D forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
}

REGISTER_BINDING(avgpool1d, register_avgpool1d);
