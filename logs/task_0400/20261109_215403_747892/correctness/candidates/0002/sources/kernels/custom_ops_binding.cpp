#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/api/include/torch/types.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu with C linkage.
extern "C" void max_reduce_launcher(
    const float* x,
    float* out_values,
    int64_t* out_indices,
    int ndim,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream);

extern "C" void relu6_launcher(const float* input, float* output, int64_t n, cudaStream_t stream);

// Forward declaration for registration
 torch::return_types::max max_reduce(torch::Tensor x, int64_t dim, bool keepdim);

// Wrapper for torch.max(x, dim=dim, keepdim=keepdim) returning torch.return_types.max.
torch::return_types::max max_reduce(torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "max_reduce: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_reduce: input must be float32");
    TORCH_CHECK(x.dim() > 0, "max_reduce: input must have at least one dimension");

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "max_reduce: dim out of range");

    // Compute output shape: same as input except dim set to 1 if keepdim, otherwise removed.
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i == dim) {
            if (keepdim) output_shape.push_back(1);
        } else {
            output_shape.push_back(x.size(i));
        }
    }

    auto values = torch::empty(output_shape, x.options());
    auto indices = torch::empty(output_shape, x.options().dtype(torch::kInt64));

    // Fill arrays for shapes and strides. Input is original x; output is contiguous as created.
    std::vector<int64_t> input_shape(x.dim());
    std::vector<int64_t> input_strides(x.dim());
    for (int64_t i = 0; i < x.dim(); ++i) {
        input_shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
    }
    std::vector<int64_t> out_shape = output_shape;
    std::vector<int64_t> out_strides(output_shape.size());
    int64_t stride = 1;
    for (int64_t i = output_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= output_shape[i];
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_reduce_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        static_cast<int>(x.dim()),
        input_shape.data(),
        input_strides.data(),
        out_shape.data(),
        out_strides.data(),
        dim,
        keepdim,
        stream);

    // Check for kernel launch errors.
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_reduce kernel launch failed: ", cudaGetErrorString(err));

    return torch::return_types::max(std::move(values), std::move(indices));
}

// Wrapper for relu6 (elementwise min(max(x,0),6)).
torch::Tensor relu6(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu6: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu6: input must be float32");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu6_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu6 kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_reduce", &max_reduce,
          "max_reduce(x, dim, keepdim) -> (values, indices)\n"
          "Computes torch.max over the given dimension.",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));

    m.def("relu6", &relu6,
          "relu6(x) -> Tensor\n"
          "Applies ReLU6 activation elementwise.",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
