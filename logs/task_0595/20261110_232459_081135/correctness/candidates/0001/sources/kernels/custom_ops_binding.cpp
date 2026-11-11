#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void custom_adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t L, int64_t output_size,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_l,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_l,
    cudaStream_t stream);

torch::Tensor custom_adaptive_avg_pool1d(torch::Tensor x, int64_t output_size) {
    // Check input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor (batch, channels, length)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output_size > 0, "output_size must be positive");
    TORCH_CHECK(output_size <= x.size(2), "output_size must not exceed input length");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    // Allocate output tensor (contiguous by default)
    auto output = torch::empty({N, C, output_size}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_adaptive_avg_pool1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, L, output_size,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        stream
    );

    // Check for kernel launch errors (optional but recommended for debugging)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_adaptive_avg_pool1d(pybind11::module& m) {
    m.def("custom_adaptive_avg_pool1d", &custom_adaptive_avg_pool1d,
          "Adaptive average pooling 1d",
          py::arg("x"),
          py::arg("output_size"));
}

REGISTER_BINDING(custom_adaptive_avg_pool1d, register_custom_adaptive_avg_pool1d);
