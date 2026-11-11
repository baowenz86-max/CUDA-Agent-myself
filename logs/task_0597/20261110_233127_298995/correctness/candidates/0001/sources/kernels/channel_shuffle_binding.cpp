#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the launcher from custom_ops.cu
extern "C" void channel_shuffle_launcher(
    float* output,
    const float* input,
    int N,
    int C,
    int H,
    int W,
    int groups,
    cudaStream_t stream);

// PyTorch wrapper for channel shuffle forward
torch::Tensor channel_shuffle_forward(torch::Tensor x, int64_t groups) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(groups > 0, "groups must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(C % groups == 0, "C must be divisible by groups");

    // Allocate output tensor of same shape
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call the launcher
    channel_shuffle_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(groups),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_channel_shuffle(pybind11::module& m) {
    m.def("channel_shuffle_forward", &channel_shuffle_forward,
          "Channel shuffle forward",
          py::arg("x"),
          py::arg("groups"));
}

// Auto-register
REGISTER_BINDING(channel_shuffle, register_channel_shuffle);
