#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void pad2d_constant_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int pad,
    float value,
    cudaStream_t stream);

extern "C" void sum_all_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

// PyTorch wrapper for constant padding on 4D tensor
torch::Tensor pad2d_constant_forward(
    torch::Tensor input,
    int64_t pad,
    double value)
{
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Extract dimensions and validate pad
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    TORCH_CHECK(pad >= 0, "pad must be non-negative");
    int H_out = H_in + 2 * static_cast<int>(pad);
    int W_out = W_in + 2 * static_cast<int>(pad);

    // Allocate output
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    pad2d_constant_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        static_cast<int>(pad),
        static_cast<float>(value),
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "pad2d_constant kernel launch failed");

    return output;
}

// PyTorch wrapper for sum of all elements
torch::Tensor sum_all_forward(torch::Tensor input)
{
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    // Allocate scalar output (0-dimensional)
    auto output = torch::empty({}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (numel as long long)
    sum_all_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(input.numel()),
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sum_all kernel launch failed");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("pad2d_constant", &pad2d_constant_forward,
          "Constant padding for 4D tensor",
          py::arg("input"),
          py::arg("pad"),
          py::arg("value"));
    m.def("sum_all", &sum_all_forward,
          "Sum all elements",
          py::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
