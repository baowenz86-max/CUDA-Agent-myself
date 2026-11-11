#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reduce_sum_dim_launcher(
    const float* input,
    float* output,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    const int64_t* out_strides,
    int ndim,
    int dim,
    long long total_output,
    cudaStream_t stream);

extern "C" void count_nonzero_dim_launcher(
    const float* input,
    int64_t* output,
    const int64_t* shape,
    const int64_t* strides,
    const int64_t* out_shape,
    const int64_t* out_strides,
    int ndim,
    int dim,
    long long total_output,
    cudaStream_t stream);

static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

static torch::Tensor reduce_sum_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > 0, "Input must have at least 1 dimension");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    // Ensure contiguous for simple stride computation
    x = x.contiguous();

    auto sizes = x.sizes();
    int ndim = x.dim();
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = sizes[i];
        strides[i] = x.strides()[i]; // strides in elements (since contiguous, but this is safe)
    }

    // Compute output shape and strides (remove dim)
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            out_shape.push_back(sizes[i]);
        }
    }

    // Allocate output tensor
    auto output = torch::empty(c10::IntArrayRef(out_shape.data(), out_shape.size()), x.options());

    // Compute output strides (contiguous)
    long long total_output = output.numel();
    std::vector<int64_t> out_strides(ndim - 1);
    if (ndim - 1 > 0) {
        out_strides[ndim - 2] = 1;
        for (int i = ndim - 3; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reduce_sum_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        shape.data(),
        strides.data(),
        out_shape.data(),
        out_strides.data(),
        ndim,
        static_cast<int>(dim),
        total_output,
        stream
    );

    check_cuda_error("reduce_sum_dim_launcher failed");
    return output;
}

static torch::Tensor count_nonzero_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() > 0, "Input must have at least 1 dimension");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    x = x.contiguous();

    auto sizes = x.sizes();
    int ndim = x.dim();
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = sizes[i];
        strides[i] = x.strides()[i];
    }

    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            out_shape.push_back(sizes[i]);
        }
    }

    auto output = torch::empty(c10::IntArrayRef(out_shape.data(), out_shape.size()), x.options().dtype(torch::kInt64));

    long long total_output = output.numel();
    std::vector<int64_t> out_strides(ndim - 1);
    if (ndim - 1 > 0) {
        out_strides[ndim - 2] = 1;
        for (int i = ndim - 3; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    count_nonzero_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        shape.data(),
        strides.data(),
        out_shape.data(),
        out_strides.data(),
        ndim,
        static_cast<int>(dim),
        total_output,
        stream
    );

    check_cuda_error("count_nonzero_dim_launcher failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reduce_sum_dim", &reduce_sum_dim_forward, "Reduce sum along a dimension",
          py::arg("x"), py::arg("dim"));
    m.def("count_nonzero_dim", &count_nonzero_dim_forward, "Count non-zero along a dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
