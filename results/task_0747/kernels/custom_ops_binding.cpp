#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_only_launcher(
    const float* input,
    float* output,
    const int* input_sizes,
    int ndim,
    int dim,
    cudaStream_t stream);

extern "C" void transpose_only_launcher(
    const float* input,
    float* output,
    const int* input_sizes,
    const int* output_sizes,
    int ndim,
    int dim0,
    int dim1,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// mean_only forward
// -----------------------------------------------------------------------------
torch::Tensor mean_only(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "mean_only: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_only: input must be float32");

    int64_t ndim64 = x.dim();
    TORCH_CHECK(ndim64 > 0, "mean_only: input must have at least one dimension");

    // Normalize dim
    int64_t norm_dim = dim < 0 ? dim + ndim64 : dim;
    TORCH_CHECK(norm_dim >= 0 && norm_dim < ndim64,
                "mean_only: dim out of range for input with ", ndim64, " dimensions");

    // Ensure contiguous input for the launcher
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // Compute output sizes (remove dimension norm_dim)
    std::vector<int64_t> output_sizes;
    output_sizes.reserve(ndim64 - 1);
    for (int64_t d = 0; d < ndim64; ++d) {
        if (d != norm_dim) output_sizes.push_back(x.size(d));
    }

    auto output = torch::empty(output_sizes, x.options());

    // Prepare input_sizes array as int (launcher expects int*)
    std::vector<int> input_sizes(ndim64);
    for (int64_t d = 0; d < ndim64; ++d) {
        input_sizes[d] = static_cast<int>(x.size(d));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_only_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_sizes.data(),
        static_cast<int>(ndim64),
        static_cast<int>(norm_dim),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_only: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// transpose_only forward
// -----------------------------------------------------------------------------
torch::Tensor transpose_only(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "transpose_only: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "transpose_only: input must be float32");

    int64_t ndim64 = x.dim();
    TORCH_CHECK(ndim64 > 0, "transpose_only: input must have at least one dimension");

    // Normalize dims
    int64_t ndim0 = dim0 < 0 ? dim0 + ndim64 : dim0;
    int64_t ndim1 = dim1 < 0 ? dim1 + ndim64 : dim1;
    TORCH_CHECK(ndim0 >= 0 && ndim0 < ndim64, "transpose_only: dim0 out of range");
    TORCH_CHECK(ndim1 >= 0 && ndim1 < ndim64, "transpose_only: dim1 out of range");
    TORCH_CHECK(ndim0 != ndim1, "transpose_only: dim0 and dim1 must be different");

    // For PyTorch view semantics, use tensor.transpose which preserves strides,
    // does not copy data, and exactly matches torch.transpose behavior.
    return x.transpose(ndim0, ndim1);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("mean_only", &mean_only, "mean_only forward",
          py::arg("x"), py::arg("dim"));
    m.def("transpose_only", &transpose_only, "transpose_only forward",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
