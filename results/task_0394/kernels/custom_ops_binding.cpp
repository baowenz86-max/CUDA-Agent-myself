#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(
    const float* x,
    float* out,
    int64_t batch,
    int64_t seq_len,
    int64_t dim_size,
    cudaStream_t stream);

extern "C" void custom_add_launcher(
    const float* x,
    const float* bias,
    float* out,
    int64_t total_elements,
    int64_t dim_size,
    cudaStream_t stream);

// Wrapper for softmax along given dim (only dim=2 supported in current kernel)
torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D (batch, seq_len, dim)");
    TORCH_CHECK(dim == 2, "Softmax dimension must be 2 for this implementation");

    // Ensure contiguity for the kernel's assumption: row-major with last dim contiguous
    auto x_contig = x.contiguous();

    auto batch = x_contig.size(0);
    auto seq_len = x_contig.size(1);
    auto dim_size = x_contig.size(2);

    // Allocate output tensor with same shape and options
    auto out = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_softmax_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        dim_size,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_launcher failed: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for elementwise add with bias broadcast
torch::Tensor custom_add_forward(torch::Tensor x, torch::Tensor bias) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Input tensor bias must be float32");
    TORCH_CHECK(x.dim() == 3, "Input tensor x must be 3D (batch, seq_len, dim)");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(2), "Bias size must match last dimension of x");

    // Ensure both are contiguous for kernel assumptions
    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    int64_t total_elements = x_contig.numel();
    int64_t dim_size = bias_contig.size(0);

    // Allocate output tensor with same shape as x
    auto out = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_add_launcher(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        dim_size,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_add_launcher failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_forward", &custom_softmax_forward,
          "Custom softmax along dimension 2 for 3D tensor",
          py::arg("x"), py::arg("dim"));
    m.def("custom_add_forward", &custom_add_forward,
          "Custom elementwise add with bias broadcast",
          py::arg("x"), py::arg("bias"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
