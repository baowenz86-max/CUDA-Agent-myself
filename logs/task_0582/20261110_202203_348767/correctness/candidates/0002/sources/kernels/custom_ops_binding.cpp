#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N, int C, int H, int W,
    int G,
    float eps,
    cudaStream_t stream
);

extern "C" void log_softmax_dim1_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream
);

// PyTorch wrapper for group_norm_forward
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1, "weight and bias must be 1D");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C == weight.size(0), "weight size must equal C");
    TORCH_CHECK(C == bias.size(0), "bias size must equal C");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    // Allocate output tensor
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(num_groups),
        static_cast<float>(eps),
        stream
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_norm_forward CUDA error: ", cudaGetErrorString(err));

    return y;
}

// PyTorch wrapper for log_softmax_dim1_forward
torch::Tensor log_softmax_dim1_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensor
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    log_softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_dim1_forward CUDA error: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
    m.def("log_softmax_dim1_forward", &log_softmax_dim1_forward,
          "LogSoftmax along dim=1 (CUDA)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
