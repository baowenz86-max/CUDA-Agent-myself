// Binding for custom_max_broadcast operator.
// Implements PyTorch tensor validation, allocation, and launcher call.
// The actual CUDA kernel is in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration matching custom_ops.cu
extern "C" void custom_max_broadcast_launcher(
    const float* x,
    const float* tensor,
    float* out,
    long long total,
    int height,
    int width,
    cudaStream_t stream);

// PyTorch wrapper: computes element-wise max with broadcasting over last two dims.
// x: input tensor of shape [batch, channels, height, width] (contiguous)
// tensor: broadcast tensor of shape [height, width] (contiguous)
// dim: ignored, present for API compatibility.
torch::Tensor custom_max_broadcast(torch::Tensor x, torch::Tensor tensor, int64_t dim) {
    // Parameter validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(tensor.is_cuda(), "tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "tensor must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(tensor.dim() == 2, "tensor must be 2D");

    // Check that tensor's shape matches last two dims of x
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    TORCH_CHECK(x.size(-2) == height && x.size(-1) == width,
                "tensor shape must match last two dims of x");

    // Ensure input and tensor are contiguous for kernel's linear indexing
    auto x_contig = x.contiguous();
    auto tensor_contig = tensor.contiguous();

    // Allocate output with same shape as input
    auto out = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_max_broadcast_launcher(
        x_contig.data_ptr<float>(),
        tensor_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<long long>(x_contig.numel()),
        static_cast<int>(height),
        static_cast<int>(width),
        stream
    );

    // Check for kernel launch errors (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_max_broadcast(pybind11::module& m) {
    m.def("custom_max_broadcast", &custom_max_broadcast,
          "Element-wise max with broadcasting over last two dimensions",
          py::arg("x"),
          py::arg("tensor"),
          py::arg("dim") = -1);
}

// Auto-register
REGISTER_BINDING(custom_max_broadcast, register_custom_max_broadcast);
