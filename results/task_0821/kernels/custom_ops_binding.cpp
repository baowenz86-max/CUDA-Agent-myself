#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

extern "C" void argsort_dim1_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Wrapper for reflection_pad2d_forward
torch::Tensor reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int OH = H + pad_top + pad_bottom;
    int OW = W + pad_left + pad_right;

    // Allocate output tensor
    auto output = torch::empty({N, C, OH, OW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for argsort_dim1_forward
torch::Tensor argsort_dim1_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor as int64, same shape
    auto output = torch::empty({N, C, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    argsort_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, C, H, W,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));
    m.def("argsort_dim1_forward", &argsort_dim1_forward,
          "Argsort along dim=1",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
