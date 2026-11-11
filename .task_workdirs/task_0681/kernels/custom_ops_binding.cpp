#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void cat_forward(
    float* output,
    const float* x,
    const float* y,
    int64_t total_elements,
    int64_t Cx,
    int64_t Cy,
    int64_t HW,
    cudaStream_t stream);

extern "C" void multiply_scalar_forward(
    float* output,
    const float* input,
    int64_t total_elements,
    double scalar,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Binding wrapper for cat_forward
torch::Tensor cat_forward_binding(torch::Tensor x, torch::Tensor y, int64_t dim) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");

    // Check contiguous
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    // Only support dim=1 as per contract
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for cat_forward");

    // Check rank and dimensions compatibility
    TORCH_CHECK(x.dim() == 4 && y.dim() == 4, "Expected 4D tensors");
    TORCH_CHECK(x.size(0) == y.size(0), "Batch size mismatch");
    TORCH_CHECK(x.size(2) == y.size(2) && x.size(3) == y.size(3), "Spatial dimensions mismatch");

    int64_t batch = x.size(0);
    int64_t Cx = x.size(1);
    int64_t Cy = y.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;
    int64_t total_elements = batch * (Cx + Cy) * HW;

    // Allocate output tensor with correct shape [batch, Cx+Cy, H, W]
    auto output = torch::empty({batch, Cx + Cy, H, W}, x.options());

    // Ensure output is contiguous (empty should be contiguous by default)
    TORCH_CHECK(output.is_contiguous(), "Output should be contiguous");

    // Get current CUDA stream
    cudaStream_t stream = getCurrentStream();

    // Call launcher
    cat_forward(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        total_elements,
        Cx,
        Cy,
        HW,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cat_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Binding wrapper for multiply_scalar_forward
torch::Tensor multiply_scalar_forward_binding(torch::Tensor input, double scalar) {
    // Check device
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    // Check contiguous
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Allocate output same shape as input
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = getCurrentStream();

    // Call launcher
    multiply_scalar_forward(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        scalar,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "multiply_scalar_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Function to register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("cat_forward", &cat_forward_binding,
          "Concatenate two CUDA float32 tensors along dim=1",
          py::arg("x"), py::arg("y"), py::arg("dim"));
    m.def("multiply_scalar_forward", &multiply_scalar_forward_binding,
          "Multiply tensor by scalar",
          py::arg("input"), py::arg("scalar"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);