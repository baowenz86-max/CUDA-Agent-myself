#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void adaptive_avg_pool2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int output_size,
    long input_stride_n, long input_stride_c, long input_stride_h, long input_stride_w,
    cudaStream_t stream);

// PyTorch wrapper for adaptive_avg_pool2d
torch::Tensor adaptive_avg_pool2d(torch::Tensor x, int64_t output_size) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor [N, C, output_size, output_size]
    auto y = torch::empty({N, C, output_size, output_size}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with strides (elements)
    adaptive_avg_pool2d_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(output_size),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "adaptive_avg_pool2d kernel launch failed");

    return y;
}

void register_adaptive_avg_pool2d(pybind11::module& m) {
    m.def("adaptive_avg_pool2d", &adaptive_avg_pool2d,
          "Adaptive average pooling to output_size x output_size",
          py::arg("x"), py::arg("output_size"));
}

REGISTER_BINDING(adaptive_avg_pool2d, register_adaptive_avg_pool2d);
