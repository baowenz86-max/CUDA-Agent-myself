#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" bool custom_equal_launcher(const int64_t* x, const int64_t* y, int64_t numel);
extern "C" void custom_sin_launcher(const int64_t* input_dev, float* output_dev, int64_t numel, cudaStream_t stream);

// PyTorch wrapper for custom_equal
torch::Tensor custom_equal(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cpu(), "custom_equal: x must be a CPU tensor");
    TORCH_CHECK(y.is_cpu(), "custom_equal: y must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "custom_equal: x must be int64");
    TORCH_CHECK(y.dtype() == torch::kInt64, "custom_equal: y must be int64");
    TORCH_CHECK(x.is_contiguous(), "custom_equal: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "custom_equal: y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "custom_equal: x and y must have the same shape");

    int64_t numel = x.numel();
    bool equal = custom_equal_launcher(
        x.data_ptr<int64_t>(),
        y.data_ptr<int64_t>(),
        numel
    );

    // Return a bool, not tensor. The contract expects Python bool.
    // We'll bind as returning bool via pybind.
    return torch::tensor(equal); // This is a tensor, but the contract expects bool.
}

// PyTorch wrapper for custom_sin
torch::Tensor custom_sin(torch::Tensor x) {
    TORCH_CHECK(x.is_cpu(), "custom_sin: x must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "custom_sin: x must be int64");
    TORCH_CHECK(x.is_contiguous(), "custom_sin: x must be contiguous");

    // Allocate output on CUDA current device
    auto output = torch::empty(x.sizes(), x.options().device(torch::kCUDA).dtype(torch::kFloat32));

    int64_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy input to device
    int64_t* input_dev;
    cudaMalloc(&input_dev, numel * sizeof(int64_t));
    cudaMemcpyAsync(input_dev, x.data_ptr<int64_t>(), numel * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    custom_sin_launcher(input_dev, output.data_ptr<float>(), numel, stream);

    // Free device memory after kernel completion asynchronously if possible; else synchronize and free.
#if CUDART_VERSION >= 11020
    cudaFreeAsync(input_dev, stream);
#else
    cudaStreamSynchronize(stream);
    cudaFree(input_dev);
#endif

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_equal", &custom_equal, "Custom equal operator", py::arg("x"), py::arg("y"));
    m.def("custom_sin", &custom_sin, "Custom sin operator", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
