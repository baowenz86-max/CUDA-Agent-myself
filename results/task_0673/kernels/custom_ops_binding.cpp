#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void cat_launcher(float* output, const float* x1, const float* x2, long long N, long long C1, long long C2, long long H, long long W, cudaStream_t stream);
extern "C" void sigmoid_launcher(const float* input, float* output, long long total_elements, cudaStream_t stream);

// Helper to check CUDA errors after kernel launches
static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// PyTorch wrapper for cat_forward
torch::Tensor cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "x2 must be contiguous");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "Inputs must be 4D tensors");
    TORCH_CHECK(x1.size(0) == x2.size(0), "Batch dimension mismatch");
    TORCH_CHECK(x1.size(2) == x2.size(2) && x1.size(3) == x2.size(3), "Spatial dimensions mismatch");

    // Extract dimensions
    long long N = x1.size(0);
    long long C1 = x1.size(1);
    long long C2 = x2.size(1);
    long long H = x1.size(2);
    long long W = x1.size(3);

    // Allocate output tensor: shape [N, C1+C2, H, W]
    auto output = torch::empty({N, C1 + C2, H, W}, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    cat_launcher(
        output.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        N, C1, C2, H, W,
        stream
    );
    check_cuda_error();

    return output;
}

// PyTorch wrapper for sigmoid_forward
torch::Tensor sigmoid_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Allocate output tensor
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sigmoid_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );
    check_cuda_error();

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("cat_forward", &cat_forward,
          "Concatenate two tensors along dim=1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Apply sigmoid elementwise",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
