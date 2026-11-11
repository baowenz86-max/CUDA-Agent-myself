// custom_ops_binding.cpp - PyTorch bindings for channel_sum and avg_pool2d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void channel_sum_launcher(float* output, const float* input,
                                     int N, int C, int H, int W,
                                     cudaStream_t stream);

extern "C" void avg_pool2d_launcher(float* output, const float* input,
                                    int N, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream);

namespace {

// Check that kernel_size, stride, padding are valid for AvgPool2d semantics:
// kernel_size and stride must be positive, padding must be non-negative.
// Also ensure input dimensions support the computed output.
inline bool valid_pool_params(int64_t kernel_size, int64_t stride, int64_t padding,
                              int64_t H, int64_t W) {
    if (kernel_size <= 0 || stride <= 0 || padding < 0) return false;
    if (H + 2 * padding < kernel_size || W + 2 * padding < kernel_size) return false;
    return true;
}

} // anonymous namespace

torch::Tensor channel_sum_forward(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "channel_sum: input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "channel_sum: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "channel_sum: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "channel_sum: input must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "channel_sum: dimensions must be positive");

    // Allocate output tensor [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    channel_sum_launcher(output.data_ptr<float>(),
                         x.data_ptr<float>(),
                         static_cast<int>(N),
                         static_cast<int>(C),
                         static_cast<int>(H),
                         static_cast<int>(W),
                         stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "channel_sum: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor avg_pool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "avg_pool2d: input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "avg_pool2d: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "avg_pool2d: input must be contiguous");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == 1, "avg_pool2d: input must be 4D (N, 1, H, W)");

    int64_t N = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(N > 0 && H > 0 && W > 0, "avg_pool2d: dimensions must be positive");
    TORCH_CHECK(valid_pool_params(kernel_size, stride, padding, H, W),
                "avg_pool2d: invalid pooling parameters");

    int64_t OH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t OW = (W + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor [N, 1, OH, OW]
    auto output = torch::empty({N, 1, OH, OW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avg_pool2d_launcher(output.data_ptr<float>(),
                        x.data_ptr<float>(),
                        static_cast<int>(N),
                        static_cast<int>(H),
                        static_cast<int>(W),
                        static_cast<int>(kernel_size),
                        static_cast<int>(stride),
                        static_cast<int>(padding),
                        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool2d: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("channel_sum", &channel_sum_forward, "Sum over channel dimension with keepdim=True",
          py::arg("x"));
    m.def("avg_pool2d", &avg_pool2d_forward, "2D average pooling",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
