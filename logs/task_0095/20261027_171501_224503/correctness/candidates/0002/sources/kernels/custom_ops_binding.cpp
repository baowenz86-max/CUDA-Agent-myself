#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void create_linspace_launcher(
    float* output,
    float start,
    float end,
    int64_t steps,
    cudaStream_t stream);

extern "C" void tensor_sub_launcher(
    float* x_inout,
    const float* y,
    int64_t x_numel,
    int64_t L,
    cudaStream_t stream);

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int C,
    int K,
    int64_t L,
    int R,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// create_linspace
// -----------------------------------------------------------------------------
static torch::Tensor create_linspace_forward(double start, double end, int64_t steps, int64_t device) {
    TORCH_CHECK(steps > 0, "steps must be positive");
    TORCH_CHECK(device >= 0, "device index must be non-negative");
    
    // Determine CUDA device and set context
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    TORCH_CHECK(err == cudaSuccess, "Failed to get CUDA device count");
    TORCH_CHECK(device < device_count, "device index out of range");
    
    // Set current device for context
    int original_device = -1;
    cudaGetDevice(&original_device);
    cudaSetDevice((int)device);
    
    // Allocate output tensor on the specified device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, (int)device));
    auto output = torch::empty({steps}, options);
    
    // Get current CUDA stream for this device
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream((int)device).stream();
    
    create_linspace_launcher(
        output.data_ptr<float>(),
        (float)start,
        (float)end,
        steps,
        stream
    );
    
    // Restore original device
    if (original_device >= 0) {
        cudaSetDevice(original_device);
    }
    
    return output;
}

// -----------------------------------------------------------------------------
// tensor_sub
// -----------------------------------------------------------------------------
static torch::Tensor tensor_sub_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N, C, L]");
    TORCH_CHECK(y.dim() == 1, "y must be 1D [L]");
    TORCH_CHECK(x.size(2) == y.size(0), "x last dim must equal y length");
    
    // In-place modification, but we return same tensor
    int64_t x_numel = x.numel();
    int64_t L = y.size(0);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tensor_sub_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x_numel,
        L,
        stream
    );
    
    return x;
}

// -----------------------------------------------------------------------------
// conv_transpose1d
// -----------------------------------------------------------------------------
static torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N, C, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_in, C_out, kernel_size]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C_out]");
    TORCH_CHECK(x.size(1) == weight.size(0), "x channel dim must match weight in_channels");
    TORCH_CHECK(weight.size(1) == bias.size(0), "weight out_channels must match bias size");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);
    int64_t K = weight.size(1);
    int64_t R = weight.size(2);
    
    // Compute output length
    int64_t L_out = (L - 1) * stride - 2 * padding + R + output_padding;
    TORCH_CHECK(L_out > 0, "invalid output length");
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto output = torch::empty({N, K, L_out}, options);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        (int)C,
        (int)K,
        L,
        (int)R,
        (int)stride,
        (int)padding,
        (int)output_padding,
        stream
    );
    
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
static void register_custom_ops(pybind11::module& m) {
    m.def("create_linspace", &create_linspace_forward,
          "Create evenly spaced values on CUDA",
          pybind11::arg("start"),
          pybind11::arg("end"),
          pybind11::arg("steps"),
          pybind11::arg("device"));
    
    m.def("tensor_sub", &tensor_sub_forward,
          "In-place broadcast subtraction x = x - y",
          pybind11::arg("x"),
          pybind11::arg("y"));
    
    m.def("conv_transpose1d", &conv_transpose1d_forward,
          "1D transposed convolution using cuDNN",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
