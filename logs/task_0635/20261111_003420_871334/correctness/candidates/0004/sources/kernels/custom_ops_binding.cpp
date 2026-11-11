// Binding file for custom reduction operators (reduce_max and reduce_std)
// Implements PyTorch tensor handling and Python registration only;
// CUDA kernels are implemented in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Forward declarations for launchers defined in custom_ops.cu
extern "C" void reduce_max_launcher(const float* input,
                                    long long numel,
                                    float* output,
                                    cudaStream_t stream);

extern "C" void reduce_std_launcher(const float* input,
                                    long long numel,
                                    float* output,
                                    cudaStream_t stream);

// -----------------------------------------------------------------------------
// PyTorch wrapper: reduce_max
// Input: x (CUDA float32 tensor, arbitrary contiguous shape)
// Returns: scalar CUDA float32 tensor containing global maximum
// -----------------------------------------------------------------------------
torch::Tensor reduce_max_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "reduce_max: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reduce_max: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "reduce_max: input must be contiguous");
    TORCH_CHECK(x.numel() > 0, "reduce_max: input must not be empty (not supported by contract)");

    // Allocate scalar output tensor on same device
    auto output = torch::empty({}, x.options());

    // Retrieve current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch reduction
    reduce_max_launcher(x.data_ptr<float>(),
                        static_cast<long long>(x.numel()),
                        output.data_ptr<float>(),
                        stream);

    // Optionally check for kernel errors (synchronization).
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_max: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// PyTorch wrapper: reduce_std
// Input: x (CUDA float32 tensor, arbitrary contiguous shape)
// Returns: scalar CUDA float32 tensor containing unbiased standard deviation
// -----------------------------------------------------------------------------
torch::Tensor reduce_std_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "reduce_std: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reduce_std: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "reduce_std: input must be contiguous");
    TORCH_CHECK(x.numel() >= 2, "reduce_std: input must have at least 2 elements");

    // Allocate scalar output tensor on same device
    auto output = torch::empty({}, x.options());

    // Retrieve current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch std reduction
    reduce_std_launcher(x.data_ptr<float>(),
                        static_cast<long long>(x.numel()),
                        output.data_ptr<float>(),
                        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_std: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// Registration function
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("reduce_max", &reduce_max_forward,
          "Compute global maximum of a CUDA float32 tensor",
          py::arg("x"));
    m.def("reduce_std", &reduce_std_forward,
          "Compute global unbiased standard deviation of a CUDA float32 tensor",
          py::arg("x"));
}

// Auto-register with the binding registry
REGISTER_BINDING(custom_ops, register_custom_ops);
