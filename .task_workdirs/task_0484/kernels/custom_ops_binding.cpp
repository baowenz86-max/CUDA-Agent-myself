#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    const int64_t* in_shape,
    const int64_t* in_strides,
    const int64_t* out_shape,
    const int64_t* out_strides,
    int64_t dim,
    int64_t ndim,
    cudaStream_t stream);

extern "C" void custom_softmax_lastdim_launcher(
    const float* input,
    float* output,
    const int64_t* in_shape,
    const int64_t* in_strides,
    int64_t ndim,
    cudaStream_t stream);

// Helper to copy int64_t shape/stride from Tensor to std::vector
static std::vector<int64_t> get_shape(const torch::Tensor& t) {
    return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

static std::vector<int64_t> get_strides(const torch::Tensor& t) {
    return std::vector<int64_t>(t.strides().begin(), t.strides().end());
}

// PyTorch wrapper for custom_mean_dim
torch::Tensor custom_mean_dim(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "custom_mean_dim: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_mean_dim: input must be float32");
    TORCH_CHECK(input.dim() > 0, "custom_mean_dim: input must have at least one dimension");
    TORCH_CHECK(dim >= -input.dim() && dim < input.dim(), "custom_mean_dim: dim out of range");

    // Normalize dim to positive (0-based)
    if (dim < 0) {
        dim += input.dim();
    }

    auto in_shape = get_shape(input);
    auto in_strides = get_strides(input);

    // Compute output shape: same as input but dim size = 1
    std::vector<int64_t> out_shape = in_shape;
    out_shape[dim] = 1;

    // Allocate output tensor with same options as input
    auto output = torch::empty(out_shape, input.options());

    auto out_strides = get_strides(output); // contiguous strides

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_dim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_shape.data(),
        in_strides.data(),
        out_shape.data(),
        out_strides.data(),
        dim,
        input.dim(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_mean_dim: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for custom_softmax_lastdim
torch::Tensor custom_softmax_lastdim(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_softmax_lastdim: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_softmax_lastdim: input must be float32");
    TORCH_CHECK(input.dim() > 0, "custom_softmax_lastdim: input must have at least one dimension");

    auto in_shape = get_shape(input);
    auto in_strides = get_strides(input);

    // Output has same shape as input
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_lastdim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_shape.data(),
        in_strides.data(),
        input.dim(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_lastdim: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Register both ops
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim", &custom_mean_dim,
          "Compute mean along a dimension with keepdim=True",
          py::arg("input"), py::arg("dim"));
    m.def("custom_softmax_lastdim", &custom_softmax_lastdim,
          "Compute softmax along the last dimension",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
