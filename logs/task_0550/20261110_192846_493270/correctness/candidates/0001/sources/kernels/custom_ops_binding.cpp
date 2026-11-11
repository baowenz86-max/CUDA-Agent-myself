#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor maxpool2d_forward_impl(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    // Compute output shape using same formula as launcher
    int64_t OH, OW;
    if (ceil_mode) {
        OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
        OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }
    TORCH_CHECK(OH > 0 && OW > 0, "Output spatial dimensions must be positive");

    auto output = torch::empty({N, C, OH, OW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    maxpool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, stride, padding, dilation, ceil_mode,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_maxpool2d_forward_impl(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward_impl,
          "2D max pooling forward (CUDA)",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("ceil_mode"));
}

// Auto-register
REGISTER_BINDING(maxpool2d_forward_impl, register_maxpool2d_forward_impl);
