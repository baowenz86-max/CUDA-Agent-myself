#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_transpose_launcher(const float* input, float* output,
                                           const int64_t* input_shape, const int64_t* input_strides,
                                           int ndim, int dim, size_t total_elements,
                                           cudaStream_t stream);

extern "C" void custom_cat_2x_launcher(const float* input, float* output,
                                         const int64_t* input_shape, const int64_t* input_strides,
                                         int ndim, size_t total_output_elements,
                                         cudaStream_t stream);

// PyTorch wrapper for custom transpose
torch::Tensor custom_transpose(torch::Tensor x, int64_t dim) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "custom_transpose: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_transpose: input must be float32");
    TORCH_CHECK(x.dim() > 0, "custom_transpose: input must have at least one dimension");
    TORCH_CHECK(dim >= -x.dim() && dim < x.dim(), "custom_transpose: dim out of range");

    // Normalize dim positive
    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "custom_transpose: dim out of range after normalization");

    // The last dimension index
    int64_t last_dim = x.dim() - 1;

    // Get input shape and strides
    auto input_sizes = x.sizes();
    auto input_strides = x.strides();

    // Compute output sizes: swap dim and last_dim
    std::vector<int64_t> output_sizes(input_sizes.begin(), input_sizes.end());
    std::swap(output_sizes[dim], output_sizes[last_dim]);

    // Compute output strides: swap corresponding strides
    std::vector<int64_t> output_strides(input_strides.begin(), input_strides.end());
    std::swap(output_strides[dim], output_strides[last_dim]);

    // Create output tensor with specified sizes and strides, no copy
    auto output = torch::empty_strided(output_sizes, output_strides, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_transpose_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_sizes.data(),
        input_strides.data(),
        static_cast<int>(x.dim()),
        static_cast<int>(dim),
        static_cast<size_t>(x.numel()),
        stream
    );

    return output;
}

// PyTorch wrapper for custom cat 2x along last dimension
torch::Tensor custom_cat_2x(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "custom_cat_2x: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_cat_2x: input must be float32");
    TORCH_CHECK(x.dim() > 0, "custom_cat_2x: input must have at least one dimension");

    // Get input shape and strides
    auto input_sizes = x.sizes();
    auto input_strides = x.strides();

    // Output shape: last dimension doubled
    std::vector<int64_t> output_sizes(input_sizes.begin(), input_sizes.end());
    output_sizes.back() *= 2;

    // Create contiguous output tensor
    auto output = torch::empty(output_sizes, x.options());
    TORCH_CHECK(output.is_contiguous(), "custom_cat_2x: output is not contiguous");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_cat_2x_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_sizes.data(),
        input_strides.data(),
        static_cast<int>(x.dim()),
        static_cast<size_t>(output.numel()),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_transpose", &custom_transpose,
          "Custom transpose along specified dimension",
          py::arg("x"), py::arg("dim"));
    m.def("custom_cat_2x", &custom_cat_2x,
          "Concatenate tensor with itself along last dimension",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
