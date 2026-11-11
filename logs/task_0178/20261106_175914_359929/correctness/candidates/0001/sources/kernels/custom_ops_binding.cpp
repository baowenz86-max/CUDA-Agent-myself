#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Extern declarations from custom_ops.cu
extern "C" void mean_dim_launcher(const float* x,
                                  float* output,
                                  int outer_size,
                                  int dim_size,
                                  int inner_size,
                                  cudaStream_t stream);

extern "C" void cat_dim_launcher(const float* in1,
                                  const float* in2,
                                  float* out,
                                  int outer_size,
                                  int dim_size1,
                                  int dim_size2,
                                  int inner_size,
                                  cudaStream_t stream);

// Helper to compute product of dims from start to end
static inline int64_t prod(const torch::IntArrayRef& sizes, int64_t start, int64_t end) {
    int64_t p = 1;
    for (int64_t i = start; i < end; ++i) {
        p *= sizes[i];
    }
    return p;
}

// mean_dim wrapper
// Signature: mean_dim(Tensor x, int dim, bool keepdim) -> Tensor
torch::Tensor mean_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "mean_dim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim: input must be float32");
    TORCH_CHECK(x.dim() > 0, "mean_dim: input must have at least one dimension");

    // Normalize negative dim
    int64_t ndim = x.dim();
    int64_t dim_norm = (dim < 0) ? dim + ndim : dim;
    TORCH_CHECK(dim_norm >= 0 && dim_norm < ndim, "mean_dim: dim out of range");

    // Make input contiguous (if not already). This ensures the kernel's indexing works.
    torch::Tensor x_contig = x.contiguous();

    // Compute output sizes: same as input but dim size = 1 if keepdim else removed
    // keepdim is assumed True from contract, but handle both for robustness
    std::vector<int64_t> out_sizes;
    if (keepdim) {
        out_sizes.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            if (i == dim_norm) {
                out_sizes.push_back(1);
            } else {
                out_sizes.push_back(x.size(i));
            }
        }
    } else {
        out_sizes.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim_norm) {
                out_sizes.push_back(x.size(i));
            }
        }
    }

    // Allocate output with same dtype/device as input
    auto output = torch::empty(out_sizes, x.options());

    // Compute sizes for launcher
    int outer_size = static_cast<int>(prod(x_contig.sizes(), 0, dim_norm));
    int dim_size = static_cast<int>(x_contig.size(dim_norm));
    int inner_size = static_cast<int>(prod(x_contig.sizes(), dim_norm + 1, ndim));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    mean_dim_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        dim_size,
        inner_size,
        stream
    );

    return output;
}

// cat_dim wrapper
// Signature: cat_dim(Tensor mean_x, Tensor x, int dim) -> Tensor
torch::Tensor cat_dim(torch::Tensor mean_x, torch::Tensor x, int64_t dim) {
    // Check device and dtype
    TORCH_CHECK(mean_x.is_cuda() && x.is_cuda(), "cat_dim: inputs must be CUDA tensors");
    TORCH_CHECK(mean_x.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32, "cat_dim: inputs must be float32");
    TORCH_CHECK(mean_x.dim() == x.dim(), "cat_dim: tensors must have same number of dimensions");
    TORCH_CHECK(mean_x.dim() > 0, "cat_dim: input must have at least one dimension");

    int64_t ndim = mean_x.dim();
    int64_t dim_norm = (dim < 0) ? dim + ndim : dim;
    TORCH_CHECK(dim_norm >= 0 && dim_norm < ndim, "cat_dim: dim out of range");

    // Check non-cat dimensions match
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim_norm) {
            TORCH_CHECK(mean_x.size(i) == x.size(i), "cat_dim: size mismatch in dimension ", i);
        }
    }

    // Make inputs contiguous
    torch::Tensor mean_x_contig = mean_x.contiguous();
    torch::Tensor x_contig = x.contiguous();

    // Compute output sizes: same as input except cat dim is sum
    std::vector<int64_t> out_sizes(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim_norm) {
            out_sizes[i] = mean_x_contig.size(i) + x_contig.size(i);
        } else {
            out_sizes[i] = mean_x_contig.size(i);
        }
    }

    auto output = torch::empty(out_sizes, mean_x.options());

    int outer_size = static_cast<int>(prod(mean_x_contig.sizes(), 0, dim_norm));
    int dim_size1 = static_cast<int>(mean_x_contig.size(dim_norm));
    int dim_size2 = static_cast<int>(x_contig.size(dim_norm));
    int inner_size = static_cast<int>(prod(mean_x_contig.sizes(), dim_norm + 1, ndim));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cat_dim_launcher(
        mean_x_contig.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        dim_size1,
        dim_size2,
        inner_size,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim", &mean_dim, "Mean along dimension with keepdim",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("cat_dim", &cat_dim, "Concatenate two tensors along a dimension",
          py::arg("mean_x"), py::arg("x"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
