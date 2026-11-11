#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void concat_forward_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int ndim,
    int64_t concat_dim,
    const int64_t* out_shape,
    const int64_t* x1_shape,
    const int64_t* x2_shape,
    const int64_t* x1_strides,
    const int64_t* x2_strides,
    int64_t out_numel,
    cudaStream_t stream
);

extern "C" void amax_forward_launcher(
    const float* x,
    float* out,
    int ndim,
    int64_t reduce_dim,
    const int64_t* in_shape,
    const int64_t* in_strides,
    const int64_t* out_shape,
    int64_t out_numel,
    cudaStream_t stream
);

static void check_cuda_float(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

static std::vector<int64_t> get_int64_shape(const torch::Tensor& t) {
    std::vector<int64_t> shape;
    shape.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        shape.push_back(t.size(i));
    }
    return shape;
}

static std::vector<int64_t> get_int64_strides(const torch::Tensor& t) {
    std::vector<int64_t> strides;
    strides.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        strides.push_back(t.stride(i));
    }
    return strides;
}

static int64_t prod(const std::vector<int64_t>& v) {
    int64_t result = 1;
    for (auto x : v) result *= x;
    return result;
}

torch::Tensor concat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    check_cuda_float(x1, "x1");
    check_cuda_float(x2, "x2");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same number of dimensions");
    TORCH_CHECK(dim >= 0 && dim < x1.dim(), "dim out of range");
    for (int64_t i = 0; i < x1.dim(); ++i) {
        if (i != dim) {
            TORCH_CHECK(x1.size(i) == x2.size(i), "x1 and x2 must match in all dims except concat dim");
        }
    }

    auto x1_shape = get_int64_shape(x1);
    auto x2_shape = get_int64_shape(x2);
    std::vector<int64_t> out_shape = x1_shape;
    out_shape[dim] += x2_shape[dim];
    int64_t out_numel = prod(out_shape);

    auto out = torch::empty(out_shape, x1.options());

    auto x1_strides = get_int64_strides(x1);
    auto x2_strides = get_int64_strides(x2);

    TORCH_CHECK(x1_strides.size() == x1.dim() && x2_strides.size() == x2.dim());

    // Allocate device memory for shape/stride arrays
    int64_t* d_out_shape;
    int64_t* d_x1_shape;
    int64_t* d_x2_shape;
    int64_t* d_x1_strides;
    int64_t* d_x2_strides;
    cudaMalloc(&d_out_shape, sizeof(int64_t) * out_shape.size());
    cudaMalloc(&d_x1_shape, sizeof(int64_t) * x1_shape.size());
    cudaMalloc(&d_x2_shape, sizeof(int64_t) * x2_shape.size());
    cudaMalloc(&d_x1_strides, sizeof(int64_t) * x1_strides.size());
    cudaMalloc(&d_x2_strides, sizeof(int64_t) * x2_strides.size());

    cudaMemcpy(d_out_shape, out_shape.data(), sizeof(int64_t) * out_shape.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1_shape, x1_shape.data(), sizeof(int64_t) * x1_shape.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2_shape, x2_shape.data(), sizeof(int64_t) * x2_shape.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1_strides, x1_strides.data(), sizeof(int64_t) * x1_strides.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2_strides, x2_strides.data(), sizeof(int64_t) * x2_strides.size(), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    concat_forward_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        static_cast<int>(x1.dim()),
        dim,
        d_out_shape,
        d_x1_shape,
        d_x2_shape,
        d_x1_strides,
        d_x2_strides,
        out_numel,
        stream
    );

    // Free device arrays
    cudaFree(d_out_shape);
    cudaFree(d_x1_shape);
    cudaFree(d_x2_shape);
    cudaFree(d_x1_strides);
    cudaFree(d_x2_strides);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in concat_forward: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor amax_forward(torch::Tensor x, int64_t dim) {
    check_cuda_float(x, "x");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    auto in_shape = get_int64_shape(x);
    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim() - 1);
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) out_shape.push_back(in_shape[i]);
    }
    int64_t out_numel = prod(out_shape);

    auto out = torch::empty(out_shape, x.options());

    auto in_strides = get_int64_strides(x);

    int64_t* d_in_shape;
    int64_t* d_in_strides;
    int64_t* d_out_shape;
    cudaMalloc(&d_in_shape, sizeof(int64_t) * in_shape.size());
    cudaMalloc(&d_in_strides, sizeof(int64_t) * in_strides.size());
    cudaMalloc(&d_out_shape, sizeof(int64_t) * out_shape.size());

    cudaMemcpy(d_in_shape, in_shape.data(), sizeof(int64_t) * in_shape.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides.data(), sizeof(int64_t) * in_strides.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape.data(), sizeof(int64_t) * out_shape.size(), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    amax_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(x.dim()),
        dim,
        d_in_shape,
        d_in_strides,
        d_out_shape,
        out_numel,
        stream
    );

    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
    cudaFree(d_out_shape);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in amax_forward: ", cudaGetErrorString(err));
    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("concat_forward", &concat_forward,
          "Concatenate two tensors along a dimension",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("amax_forward", &amax_forward,
          "Compute maximum along a dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
