#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declaration of the launcher exported from custom_ops.cu
extern "C" void relu_trunc_forward_launcher(const float* input,
                                            float* output,
                                            long long numel,
                                            cudaStream_t stream);

// PyTorch wrapper for the fused relu+trunc operation
torch::Tensor relu_trunc_forward(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must have dtype float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must have at least one element");

    // The launcher assumes contiguous memory; make contiguous if needed.
    // This matches the contract "supports strides" by internally handling non-contiguous inputs.
    auto x_contig = x.contiguous();

    // Allocate output tensor with same shape, dtype, and device as input.
    auto output = torch::empty_like(x_contig);

    // Get the current CUDA stream.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Convert numel to long long (same as the launcher signature) and call.
    long long numel = static_cast<long long>(x_contig.numel());
    relu_trunc_forward_launcher(x_contig.data_ptr<float>(),
                                output.data_ptr<float>(),
                                numel,
                                stream);

    // Check for CUDA errors after launch (asynchronously).
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function for pybind
void register_relu_trunc_forward(pybind11::module& m) {
    m.def("relu_trunc_forward", &relu_trunc_forward,
          "Fused ReLU + truncation forward pass",
          pybind11::arg("x"));
}

// Auto-register with the binding registry
REGISTER_BINDING(relu_trunc_forward, register_relu_trunc_forward);
