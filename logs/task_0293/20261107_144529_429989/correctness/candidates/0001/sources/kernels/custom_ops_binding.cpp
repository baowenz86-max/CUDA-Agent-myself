// Binding for instance_norm2d_forward and log_softmax_dim1_forward
// No CUDA kernels here; launchers are defined in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void instance_norm2d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    float eps,
    cudaStream_t stream
);

extern "C" void log_softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream
);

// Wrapper for instance_norm2d_forward
torch::Tensor instance_norm2d_forward(
    torch::Tensor input,
    int64_t num_channels,
    double eps
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "instance_norm2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "instance_norm2d_forward: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "instance_norm2d_forward: input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "instance_norm2d_forward: input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK(C == num_channels, "instance_norm2d_forward: num_channels must match input.size(1)");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    instance_norm2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<float>(eps),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "instance_norm2d_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for log_softmax_dim1_forward
torch::Tensor log_softmax_dim1_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "log_softmax_dim1_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "log_softmax_dim1_forward: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "log_softmax_dim1_forward: input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "log_softmax_dim1_forward: input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_dim1_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_dim1_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm2d_forward", &instance_norm2d_forward,
          "InstanceNorm2d forward (affine=False, track_running_stats=False)",
          pybind11::arg("input"),
          pybind11::arg("num_channels"),
          pybind11::arg("eps"));
    m.def("log_softmax_dim1_forward", &log_softmax_dim1_forward,
          "LogSoftmax along dim=1",
          pybind11::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
