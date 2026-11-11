// PyTorch binding for custom softplus and avgpool2d CUDA kernels.
// This file contains only tensor wrapper and pybind registration;
// CUDA kernel implementations are in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softplus_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void avgpool2d_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t OH,
    int64_t OW,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream);

// Utility to ensure the tensor is contiguous and on CUDA, float32.
// Returns a contiguous tensor (may be the original if already contiguous).
static torch::Tensor ensure_contiguous(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    if (!input.is_contiguous()) {
        return input.contiguous();
    }
    return input;
}

// softplus_forward: element-wise Softplus with PyTorch default threshold=20.
torch::Tensor softplus_forward(torch::Tensor input) {
    auto contiguous_input = ensure_contiguous(input);
    auto output = torch::empty_like(contiguous_input);

    int64_t numel = contiguous_input.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(
        contiguous_input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    // Check for launch errors (optional, helps debugging).
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softplus_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// avgpool2d_forward: 2D average pooling with specified kernel_size, stride, padding.
// Input shape: [N, C, H, W], output shape computed as per PyTorch AvgPool2d.
torch::Tensor avgpool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {

    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D (N, C, H, W)");

    auto contiguous_input = input.is_contiguous() ? input : input.contiguous();

    int64_t N = contiguous_input.size(0);
    int64_t C = contiguous_input.size(1);
    int64_t H = contiguous_input.size(2);
    int64_t W = contiguous_input.size(3);

    // Basic validation: kernel_size/stride/padding should be >= 0
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Compute output spatial dimensions
    // Formula: floor((input_size + 2*padding - kernel_size) / stride) + 1
    int64_t OH = ((H + 2 * padding - kernel_size) / stride) + 1;
    int64_t OW = ((W + 2 * padding - kernel_size) / stride) + 1;

    TORCH_CHECK(OH > 0 && OW > 0, "Output dimensions must be positive");

    auto output = torch::empty({N, C, OH, OW}, contiguous_input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avgpool2d_launcher(
        contiguous_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        OH, OW,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avgpool2d_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    using namespace pybind11::literals;
    m.def("softplus_forward", &softplus_forward,
          "Compute Softplus (beta=1, threshold=20) elementwise",
          py::arg("input"));
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "Compute 2D average pooling with given kernel_size, stride, padding",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
