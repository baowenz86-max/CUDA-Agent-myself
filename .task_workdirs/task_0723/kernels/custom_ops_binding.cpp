#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu
extern "C" void float_avg_pool2d_launcher(const float* input, float* output,
                                          int64_t N, int64_t C, int64_t H, int64_t W,
                                          int64_t kernel_size, int64_t stride, int64_t padding,
                                          cudaStream_t stream);

// PyTorch binding function
static torch::Tensor avg_pool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Compute output dimensions (same as PyTorch AvgPool2d with ceil_mode=False)
    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "invalid output dimensions");

    // Allocate output tensor
    auto output = torch::empty({N, C, outH, outW}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    float_avg_pool2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, stride, padding,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_avg_pool2d(pybind11::module& m) {
    m.def("avg_pool2d_forward", &avg_pool2d_forward,
          "Average pooling 2D forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

// Auto-register
REGISTER_BINDING(avg_pool2d, register_avg_pool2d);
