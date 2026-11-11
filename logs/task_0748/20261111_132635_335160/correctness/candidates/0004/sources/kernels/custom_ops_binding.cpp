#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_softmax_launcher(float* out, const float* in, int rows, int cols, cudaStream_t stream);

// Wrapper for custom softmax (column-wise softmax)
torch::Tensor custom_softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    // The existing kernel only supports dim=0 (column-wise softmax)
    TORCH_CHECK(dim == 0, "Only dim=0 supported");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_launcher(output.data_ptr<float>(), input.data_ptr<float>(), rows, cols, stream);

    return output;
}

// Wrapper for custom unsqueeze (view)
torch::Tensor custom_unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    // Manually construct a view with an inserted dimension of size 1.
    // This avoids calling any torch:: operation that might be misinterpreted.
    auto sizes = input.sizes().vec();
    auto strides = input.strides().vec();
    int64_t normalized_dim = dim;
    if (normalized_dim < 0) normalized_dim += input.dim() + 1;
    TORCH_CHECK(normalized_dim >= 0 && normalized_dim <= input.dim(), "dim out of range");

    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i <= input.dim(); ++i) {
        if (i < normalized_dim) {
            new_sizes.push_back(sizes[i]);
            new_strides.push_back(strides[i]);
        } else if (i == normalized_dim) {
            new_sizes.push_back(1);
            // stride for the new dimension: if inserted at end, stride is 1;
            // otherwise stride equals the product of remaining strides from i onward.
            int64_t stride_val = 1;
            for (int64_t j = i; j < input.dim(); ++j) {
                stride_val *= strides[j];
            }
            new_strides.push_back(stride_val);
        } else {
            new_sizes.push_back(sizes[i - 1]);
            new_strides.push_back(strides[i - 1]);
        }
    }

    return input.as_strided(new_sizes, new_strides);
}

// Registration function
void register_custom_ops_binding(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax_forward, "Column-wise softmax",
          pybind11::arg("input"), pybind11::arg("dim"));
    m.def("custom_unsqueeze", &custom_unsqueeze_forward, "Unsqueeze view",
          pybind11::arg("input"), pybind11::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);