#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_sum_dim_launcher(
    const float* input,
    float* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t dim,
    const int64_t* output_strides,
    int64_t total_output,
    cudaStream_t stream);

extern "C" void custom_expm1_launcher(
    const float* input,
    float* output,
    int64_t total,
    cudaStream_t stream);

// Wrapper for custom_sum_dim
torch::Tensor custom_sum_dim(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4-dimensional");

    int64_t D = x.dim();
    // Normalize dim (support negative)
    if (dim < 0) dim += D;
    TORCH_CHECK(dim >= 0 && dim < D, "dim out of range");

    // Compute output shape by removing dimension 'dim'
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < D; ++i) {
        if (i != dim) output_shape.push_back(x.size(i));
    }

    auto output = torch::empty(output_shape, x.options());

    // Prepare input shape and strides as contiguous arrays for launcher
    std::vector<int64_t> input_shape(D);
    std::vector<int64_t> input_strides(D);
    for (int64_t i = 0; i < D; ++i) {
        input_shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
    }

    // Output strides (contiguous)
    std::vector<int64_t> output_strides(output_shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(output_shape.size()) - 1; i >= 0; --i) {
        output_strides[i] = stride;
        stride *= output_shape[i];
    }

    int64_t total_output = output.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_sum_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_shape.data(),
        input_strides.data(),
        dim,
        output_strides.data(),
        total_output,
        stream);

    // Check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sum_dim kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for custom_expm1
torch::Tensor custom_expm1(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto output = torch::empty_like(x);

    int64_t total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_expm1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    // Check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_expm1 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_sum_dim", &custom_sum_dim,
          "Sum over a specified dimension of a 4D float32 CUDA tensor",
          py::arg("x"), py::arg("dim"));
    m.def("custom_expm1", &custom_expm1,
          "Elementwise expm1 for float32 CUDA tensor",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
