// PyTorch binding for custom log_softmax and mul_scalar kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_log_softmax_launcher(
    const float* input,
    float* output,
    int B,
    int H,
    int W,
    cudaStream_t stream
);

extern "C" void custom_mul_scalar_launcher(
    const float* input,
    float* output,
    float scalar,
    int total_elements,
    cudaStream_t stream
);

// Wrapper for custom_log_softmax
torch::Tensor custom_log_softmax(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    // Ensure contiguous (if not, make contiguous copy)
    auto x_contig = x.contiguous();

    int B = x_contig.size(0);
    int H = x_contig.size(1);
    int W = x_contig.size(2);

    // Allocate output tensor with same shape and options as contiguous input
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_log_softmax_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        H,
        W,
        stream
    );

    return output;
}

// Wrapper for custom_mul_scalar
torch::Tensor custom_mul_scalar(torch::Tensor x, double scalar) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous
    auto x_contig = x.contiguous();

    // Allocate output
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_mul_scalar_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scalar),
        x_contig.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_log_softmax", &custom_log_softmax,
          "LogSoftmax along dim=1 for 3D tensor",
          pybind11::arg("x"));
    m.def("custom_mul_scalar", &custom_mul_scalar,
          "Multiply tensor by scalar",
          pybind11::arg("x"), pybind11::arg("scalar"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
