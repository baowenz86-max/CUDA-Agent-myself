// Binding file for sigmoid and softmax_dim1 custom operations
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream);

// Wrapper for sigmoid
static torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t numel = x.numel();

    sigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream);

    return output;
}

// Wrapper for softmax over dim=1
static torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    auto output = torch::empty_like(x);

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return output;
}

// Register both functions
static void register_custom_ops(py::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward (CUDA)");
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax dim=1 forward (CUDA)");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
