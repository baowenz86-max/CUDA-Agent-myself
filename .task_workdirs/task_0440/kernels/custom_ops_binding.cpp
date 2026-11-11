#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void instance_norm_2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream);

extern "C" void relu_launcher(
    const float* input,
    float* output,
    size_t total,
    cudaStream_t stream);

static torch::Tensor instance_norm_2d_forward(torch::Tensor input, double eps = 1e-5) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    instance_norm_2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps),
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

static torch::Tensor relu_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    const size_t total = input.numel();
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    relu_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm_2d_forward", &instance_norm_2d_forward,
          "InstanceNorm2d forward (affine=False, track_running_stats=False)",
          pybind11::arg("input"),
          pybind11::arg("eps") = 1e-5);
    m.def("relu_forward", &relu_forward,
          "ReLU activation forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);