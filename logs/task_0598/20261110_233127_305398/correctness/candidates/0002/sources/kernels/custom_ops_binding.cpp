#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_cat_launcher(
    const float* x1_ptr,
    const float* x2_ptr,
    float* out_ptr,
    int64_t ndim,
    const int64_t* x1_shape,
    const int64_t* x2_shape,
    const int64_t* x1_strides,
    const int64_t* x2_strides,
    const int64_t* out_strides,
    int64_t dim,
    int64_t total_output_elems,
    cudaStream_t stream
);

torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Input validation
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same number of dimensions");
    int64_t ndim = x1.dim();
    TORCH_CHECK(ndim > 0, "cannot concatenate 0-dimensional tensors");
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Check shapes match except in dim
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            TORCH_CHECK(x1.size(d) == x2.size(d), "x1 and x2 must have same shape except in concatenation dimension");
        }
    }

    // Make contiguous copies if needed (handles arbitrary strides)
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    // Compute output shape
    std::vector<int64_t> out_shape(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        out_shape[d] = x1_contig.size(d);
    }
    out_shape[dim] = x1_contig.size(dim) + x2_contig.size(dim);

    auto out = torch::empty(out_shape, x1_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get raw pointers and metadata
    const float* x1_ptr = x1_contig.data_ptr<float>();
    const float* x2_ptr = x2_contig.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    // We need shape and stride arrays as int64_t arrays for the launcher
    std::vector<int64_t> x1_shape_vec(ndim);
    std::vector<int64_t> x2_shape_vec(ndim);
    std::vector<int64_t> x1_strides_vec(ndim);
    std::vector<int64_t> x2_strides_vec(ndim);
    std::vector<int64_t> out_strides_vec(ndim);

    for (int64_t d = 0; d < ndim; ++d) {
        x1_shape_vec[d] = x1_contig.size(d);
        x2_shape_vec[d] = x2_contig.size(d);
        x1_strides_vec[d] = x1_contig.stride(d);
        x2_strides_vec[d] = x2_contig.stride(d);
        out_strides_vec[d] = out.stride(d);
    }

    int64_t total_output_elems = out.numel();

    custom_cat_launcher(
        x1_ptr,
        x2_ptr,
        out_ptr,
        ndim,
        x1_shape_vec.data(),
        x2_shape_vec.data(),
        x1_strides_vec.data(),
        x2_strides_vec.data(),
        out_strides_vec.data(),
        dim,
        total_output_elems,
        stream
    );

    return out;
}

void register_custom_cat(pybind11::module& m) {
    m.def("custom_cat", &custom_cat,
          "Concatenate two tensors along a dimension",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
}

REGISTER_BINDING(custom_cat, register_custom_cat);
