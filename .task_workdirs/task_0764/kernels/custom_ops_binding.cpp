#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_over_dim0_launcher(
    const float* x,
    float* output,
    int64_t dim0_size,
    int64_t inner_size,
    cudaStream_t stream
);

extern "C" void softmax_over_dim0_launcher(
    const float* x,
    float* output,
    int64_t dim0_size,
    int64_t inner_size,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// PyTorch wrapper for sum over dimension 0
torch::Tensor sum_over_dim0(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > dim, "dim out of range");

    // Make contiguous to ensure launcher assumption; preserve original if already contiguous
    auto x_contig = x.contiguous();

    int64_t dim0_size = x_contig.size(dim);
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < x_contig.dim(); ++d) {
        inner_size *= x_contig.size(d);
    }

    // Output shape: remove dimension 'dim'
    std::vector<int64_t> output_sizes;
    for (int64_t d = 0; d < x_contig.dim(); ++d) {
        if (d != dim) {
            output_sizes.push_back(x_contig.size(d));
        }
    }
    auto output = torch::empty(output_sizes, x_contig.options());

    if (dim0_size > 0 && inner_size > 0) {
        cudaStream_t stream = get_current_stream();
        sum_over_dim0_launcher(
            x_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            dim0_size,
            inner_size,
            stream
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

// PyTorch wrapper for softmax over dimension 0
torch::Tensor softmax_over_dim0(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > dim, "dim out of range");

    auto x_contig = x.contiguous();

    int64_t dim0_size = x_contig.size(dim);
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < x_contig.dim(); ++d) {
        inner_size *= x_contig.size(d);
    }

    // Output same shape as input
    auto output = torch::empty_like(x_contig);

    if (dim0_size > 0 && inner_size > 0) {
        cudaStream_t stream = get_current_stream();
        softmax_over_dim0_launcher(
            x_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            dim0_size,
            inner_size,
            stream
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("sum_over_dim0", &sum_over_dim0,
          "Sum over dimension 0",
          py::arg("x"), py::arg("dim"));
    m.def("softmax_over_dim0", &softmax_over_dim0,
          "Softmax over dimension 0",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
