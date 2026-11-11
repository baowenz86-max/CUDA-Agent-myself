#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void concat_dim_launcher(
    float* out,
    const float* x1,
    const float* x2,
    long long total_elems,
    int dim,
    int rank,
    const long long* out_shape,
    const long long* x1_strides,
    const long long* x2_strides,
    const long long* out_strides,
    cudaStream_t stream
);

// No launcher needed for rot90 because it's a view operation; no custom CUDA kernel required.

torch::Tensor concat_dim(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same rank");
    int rank = x1.dim();
    TORCH_CHECK(dim >= 0 && dim < rank, "dim out of range");
    for (int i = 0; i < rank; ++i) {
        if (i != dim) {
            TORCH_CHECK(x1.size(i) == x2.size(i), "x1 and x2 must have same size except along dim");
        }
    }
    // Ensure inputs are contiguous? The kernel can handle strides, but we pass strides to launcher.
    // For simplicity, we can make them contiguous to avoid issues; but we want to support strided inputs if possible.
    // However, the launcher expects pointers and stride arrays; it can handle strided tensors.
    // We'll keep them as is.
    
    // Compute output shape
    std::vector<int64_t> out_sizes(rank);
    for (int i = 0; i < rank; ++i) {
        out_sizes[i] = (i == dim) ? x1.size(i) + x2.size(i) : x1.size(i);
    }
    auto out = torch::empty(out_sizes, x1.options()); // contiguous by default
    
    // Prepare host-side arrays for shapes and strides (we pass host pointers; launcher will copy to device)
    // Actually, the launcher expects host pointers? Looking at custom_ops.cu, the launcher copies them to device using cudaMemcpyAsync.
    // So we can pass host arrays.
    std::vector<long long> out_shape(rank), x1_strides(rank), x2_strides(rank), out_strides(rank);
    for (int i = 0; i < rank; ++i) {
        out_shape[i] = out.size(i);
        x1_strides[i] = x1.stride(i);
        x2_strides[i] = x2.stride(i);
        out_strides[i] = out.stride(i);
    }
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    concat_dim_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.numel(),
        static_cast<int>(dim),
        rank,
        out_shape.data(),
        x1_strides.data(),
        x2_strides.data(),
        out_strides.data(),
        stream
    );
    
    return out;
}

torch::Tensor rot90_k1_dims23(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "rot90_k1_dims23 requires 4D tensor");
    
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    
    // For rot90 with k=1, dims=(2,3): output shape is [B, C, W, H]
    // and strides become (s0, s1, s3, s2) where s0,s1,s2,s3 are input strides.
    auto s0 = x.stride(0);
    auto s1 = x.stride(1);
    auto s2 = x.stride(2);
    auto s3 = x.stride(3);
    
    std::vector<int64_t> sizes = {B, C, W, H};
    std::vector<int64_t> strides = {s0, s1, s3, s2};
    
    // Return a view (no copy)
    return x.as_strided(sizes, strides);
}

void register_custom_ops(pybind11::module& m) {
    m.def("concat_dim", &concat_dim, "Concatenate two tensors along a dimension",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("rot90_k1_dims23", &rot90_k1_dims23, "Rotate tensor 90 degrees in dims 2 and 3 (view)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);