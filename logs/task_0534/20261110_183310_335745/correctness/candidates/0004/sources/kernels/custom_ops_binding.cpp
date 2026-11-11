// Binding for max_dim_keepdim_forward and cosine_similarity_dim_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

#include <vector>
#include <cstdint>

// Declare launchers from custom_ops.cu
extern "C" void max_dim_keepdim_launcher(
    const float* input,
    float* output_values,
    int64_t* output_indices,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream);

extern "C" void cosine_similarity_dim_launcher(
    const float* x1,
    const float* x2,
    float* output,
    const int64_t* x1_shape,
    const int64_t* x1_strides,
    const int64_t* x2_shape,
    const int64_t* x2_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    cudaStream_t stream);

// Helper: normalize negative dim and validate range
static int64_t normalize_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range (expected to be in range of [-", ndim, ", ", ndim - 1, "], but got ", dim);
    return dim;
}

// Helper: compute output shape for reduction along dim with keepdim
static std::vector<int64_t> compute_output_shape(const std::vector<int64_t>& input_shape, int64_t dim, bool keepdim) {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
        if (i == dim) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(input_shape[i]);
        }
    }
    return out;
}

// Helper: compute strides for a contiguous tensor of given shape
static std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t acc = 1;
    for (int i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

// max_dim_keepdim_forward implementation
static std::tuple<torch::Tensor, torch::Tensor> max_dim_keepdim_forward_impl(
    torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");

    int64_t ndim = x.dim();
    dim = normalize_dim(dim, ndim);

    auto input_sizes = x.sizes().vec();
    auto input_strides = x.strides().vec();

    auto output_shape_vec = compute_output_shape(input_sizes, dim, keepdim);
    auto output_strides_vec = compute_contiguous_strides(output_shape_vec);

    auto options_values = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto options_indices = torch::TensorOptions().dtype(torch::kInt64).device(x.device());

    torch::Tensor output_values = torch::empty(output_shape_vec, options_values);
    torch::Tensor output_indices = torch::empty(output_shape_vec, options_indices);

    std::vector<int64_t> input_shape_i64(input_sizes.begin(), input_sizes.end());
    std::vector<int64_t> input_strides_i64(input_strides.begin(), input_strides.end());
    std::vector<int64_t> output_shape_i64(output_shape_vec.begin(), output_shape_vec.end());
    std::vector<int64_t> output_strides_i64(output_strides_vec.begin(), output_strides_vec.end());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_dim_keepdim_launcher(
        x.data_ptr<float>(),
        output_values.data_ptr<float>(),
        output_indices.data_ptr<int64_t>(),
        input_shape_i64.data(),
        input_strides_i64.data(),
        output_shape_i64.data(),
        output_strides_i64.data(),
        static_cast<int>(ndim),
        dim,
        keepdim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed in max_dim_keepdim: ", cudaGetErrorString(err));

    return std::make_tuple(output_values, output_indices);
}

// cosine_similarity_dim_forward implementation
static torch::Tensor cosine_similarity_dim_forward_impl(
    torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be CUDA");
    TORCH_CHECK(x2.is_cuda(), "x2 must be CUDA");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have the same number of dimensions");
    TORCH_CHECK(x1.dim() > 0, "Input tensors must have at least one dimension");

    int64_t ndim = x1.dim();
    dim = normalize_dim(dim, ndim);

    auto x1_sizes = x1.sizes().vec();
    auto x2_sizes = x2.sizes().vec();

    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        TORCH_CHECK(x1_sizes[i] == x2_sizes[i] || x2_sizes[i] == 1,
                    "Broadcasting dimension mismatch at dim ", i);
    }
    TORCH_CHECK(x1_sizes[dim] == x2_sizes[dim] || x2_sizes[dim] == 1,
                "Broadcasting dimension mismatch at reduced dim ", dim);

    std::vector<int64_t> output_shape_vec;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) output_shape_vec.push_back(x1_sizes[i]);
    }
    auto output_strides_vec = compute_contiguous_strides(output_shape_vec);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x1.device());
    torch::Tensor output = torch::empty(output_shape_vec, options);

    std::vector<int64_t> x1_shape_i64(x1_sizes.begin(), x1_sizes.end());
    std::vector<int64_t> x1_strides_i64(x1.strides().vec().begin(), x1.strides().vec().end());
    std::vector<int64_t> x2_shape_i64(x2_sizes.begin(), x2_sizes.end());
    std::vector<int64_t> x2_strides_i64(x2.strides().vec().begin(), x2.strides().vec().end());
    std::vector<int64_t> output_shape_i64(output_shape_vec.begin(), output_shape_vec.end());
    std::vector<int64_t> output_strides_i64(output_strides_vec.begin(), output_strides_vec.end());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cosine_similarity_dim_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        output.data_ptr<float>(),
        x1_shape_i64.data(),
        x1_strides_i64.data(),
        x2_shape_i64.data(),
        x2_strides_i64.data(),
        output_shape_i64.data(),
        output_strides_i64.data(),
        static_cast<int>(ndim),
        dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed in cosine_similarity: ", cudaGetErrorString(err));

    return output;
}

// Python-facing functions
static std::tuple<torch::Tensor, torch::Tensor> max_dim_keepdim_forward(
    torch::Tensor x, int64_t dim, bool keepdim) {
    return max_dim_keepdim_forward_impl(x, dim, keepdim);
}

static torch::Tensor cosine_similarity_dim_forward(
    torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    return cosine_similarity_dim_forward_impl(x1, x2, dim);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("max_dim_keepdim_forward", &max_dim_keepdim_forward,
          "Compute max along dim with keepdim, returning (values, indices)",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("cosine_similarity_dim_forward", &cosine_similarity_dim_forward,
          "Compute cosine similarity along dim",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
