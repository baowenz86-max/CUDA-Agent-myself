// Binding for custom ops: linspace_1d, elementwise_sub, conv_transpose1d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t linspace_1d_launcher(
    float* output, float start, float end, int steps, cudaStream_t stream);

extern "C" cudaError_t elementwise_sub_launcher(
    const float* x, const float* y, float* out,
    int batch, int channels, int length,
    int x_batch_stride, int x_channel_stride, int y_stride,
    cudaStream_t stream);

extern "C" cudaError_t conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_length,
    cudaStream_t stream);

// Helper to check CUDA errors
static void checkCudaError(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Helper to get current CUDA stream
static cudaStream_t getCurrentCudaStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// linspace_1d binding
torch::Tensor linspace_1d_forward(double start, double end, int64_t steps, int64_t device_index) {
    TORCH_CHECK(steps >= 0, "steps must be non-negative, got ", steps);
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative, got ", device_index);

    // Create tensor on specified device using empty
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, device_index));
    auto output = torch::empty({steps}, options);

    if (steps == 0) {
        return output;
    }

    cudaStream_t stream = getCurrentCudaStream();
    cudaError_t err = linspace_1d_launcher(
        output.data_ptr<float>(),
        static_cast<float>(start),
        static_cast<float>(end),
        static_cast<int>(steps),
        stream
    );
    checkCudaError(err, "linspace_1d_launcher failed");
    return output;
}

// elementwise_sub binding
torch::Tensor elementwise_sub_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (length)");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t length = x.size(2);
    TORCH_CHECK(y.size(0) == length, "y length (", y.size(0), ") must match x last dim (", length, ")");

    // Ensure contiguous for kernel; if not, we can use strides but kernel supports strides
    // For output, allocate contiguous tensor of same shape
    auto output = torch::empty_like(x); // preserves shape, dtype, device, but will be contiguous by default

    cudaStream_t stream = getCurrentCudaStream();

    // Get strides in elements
    int64_t x_batch_stride = x.stride(0);
    int64_t x_channel_stride = x.stride(1);
    int64_t y_stride = y.stride(0);

    cudaError_t err = elementwise_sub_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(channels),
        static_cast<int>(length),
        static_cast<int>(x_batch_stride),
        static_cast<int>(x_channel_stride),
        static_cast<int>(y_stride),
        stream
    );
    checkCudaError(err, "elementwise_sub_launcher failed");
    return output;
}

// conv_transpose1d binding using custom CUDA kernel
torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor or None");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32 if provided");
    }
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    if (bias.defined()) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    }

    int64_t batch_size = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t input_length = x.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    // Validate weight shape matches in_channels
    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels must match x in_channels");
    if (bias.defined()) {
        TORCH_CHECK(bias.size(0) == out_channels, "bias size must match out_channels");
    }

    // Compute output length
    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive, got ", output_length);

    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto output = torch::empty({batch_size, out_channels, output_length}, options);

    // If output is empty (batch or channels zero), return empty
    if (batch_size == 0 || out_channels == 0 || output_length == 0) {
        return output;
    }

    // Get current stream
    cudaStream_t stream = getCurrentCudaStream();

    // Call custom CUDA launcher
    cudaError_t err = conv_transpose1d_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(input_length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_length),
        stream
    );
    checkCudaError(err, "conv_transpose1d_launcher failed");

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("linspace_1d", &linspace_1d_forward,
          "Generate linear space on CUDA",
          pybind11::arg("start"),
          pybind11::arg("end"),
          pybind11::arg("steps"),
          pybind11::arg("device"));
    m.def("elementwise_sub", &elementwise_sub_forward,
          "Elementwise subtraction with broadcasting along last dimension",
          pybind11::arg("x"),
          pybind11::arg("y"));
    m.def("conv_transpose1d", &conv_transpose1d_forward,
          "1D transposed convolution using custom CUDA kernel",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
