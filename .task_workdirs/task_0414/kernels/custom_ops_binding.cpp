// Binding for custom bicubic interpolation CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void interpolate_bicubic_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    double scale_factor,
    cudaStream_t stream);

// PyTorch wrapper function
static torch::Tensor interpolate_bicubic_forward(
    torch::Tensor input,
    double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Compute output dimensions
    int64_t outH64 = static_cast<int64_t>(std::floor(H * scale_factor + 1e-8));
    int64_t outW64 = static_cast<int64_t>(std::floor(W * scale_factor + 1e-8));
    TORCH_CHECK(outH64 > 0 && outW64 > 0, "Output dimensions must be positive");
    TORCH_CHECK(outH64 <= INT_MAX && outW64 <= INT_MAX, "Output dimensions too large");
    int outH = static_cast<int>(outH64);
    int outW = static_cast<int>(outW64);

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    interpolate_bicubic_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        scale_factor,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_interpolate_bicubic(pybind11::module& m) {
    m.def("interpolate_bicubic_forward",
          &interpolate_bicubic_forward,
          "Bicubic interpolation forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("scale_factor"));
}

REGISTER_BINDING(interpolate_bicubic, register_interpolate_bicubic);
