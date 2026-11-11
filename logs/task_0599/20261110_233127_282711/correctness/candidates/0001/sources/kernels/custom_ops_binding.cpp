#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t ndim,
    const int64_t* out_strides,
    const int64_t* in_strides,
    const int64_t* shape,
    cudaStream_t stream);

extern "C" void add_scalar_tensor_forward_launcher(
    float* output,
    const float* x,
    const float* param,
    int64_t numel,
    int64_t ndim,
    int64_t param_ndim,
    const int64_t* out_strides,
    const int64_t* x_strides,
    const int64_t* param_strides,
    const int64_t* shape,
    cudaStream_t stream);

namespace {

std::vector<int64_t> get_shape(const torch::Tensor& t) {
    std::vector<int64_t> shape;
    shape.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        shape.push_back(t.size(i));
    }
    return shape;
}

std::vector<int64_t> get_strides(const torch::Tensor& t) {
    std::vector<int64_t> strides;
    strides.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        strides.push_back(t.stride(i));
    }
    return strides;
}

} // namespace

torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "abs_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "abs_forward: x must be float32");
    TORCH_CHECK(x.numel() > 0, "abs_forward: x must have at least one element");

    auto shape = get_shape(x);
    auto strides = get_strides(x);
    int64_t ndim = x.dim();
    int64_t numel = x.numel();

    auto output = torch::empty_strided(shape, strides, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        ndim,
        strides.data(),
        strides.data(), // input strides same as output strides (same shape)
        shape.data(),
        stream);

    return output;
}

torch::Tensor add_scalar_tensor_forward(torch::Tensor x, torch::Tensor param) {
    TORCH_CHECK(x.is_cuda(), "add_scalar_tensor_forward: x must be a CUDA tensor");
    TORCH_CHECK(param.is_cuda(), "add_scalar_tensor_forward: param must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "add_scalar_tensor_forward: x must be float32");
    TORCH_CHECK(param.dtype() == torch::kFloat32, "add_scalar_tensor_forward: param must be float32");
    TORCH_CHECK(x.numel() > 0, "add_scalar_tensor_forward: x must have at least one element");

    // Broadcast param of shape [3,256,256] over batch dimension 512.
    // Contract: x shape [512,3,256,256], param shape [3,256,256].
    // General broadcast: param dims must match trailing dims of x.
    TORCH_CHECK(param.dim() <= x.dim(), "add_scalar_tensor_forward: param must have <= dims than x");
    int64_t offset = x.dim() - param.dim();
    for (int64_t i = 0; i < param.dim(); ++i) {
        TORCH_CHECK(param.size(i) == x.size(offset + i),
                    "add_scalar_tensor_forward: param shape incompatible with x for broadcasting");
    }

    auto shape = get_shape(x);
    auto strides = get_strides(x);
    auto param_shape = get_shape(param);
    auto param_strides = get_strides(param);
    int64_t ndim = x.dim();
    int64_t param_ndim = param.dim();
    int64_t numel = x.numel();

    auto output = torch::empty_strided(shape, strides, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_scalar_tensor_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        param.data_ptr<float>(),
        numel,
        ndim,
        param_ndim,
        strides.data(),
        strides.data(),
        param_strides.data(),
        shape.data(),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("add_scalar_tensor_forward", &add_scalar_tensor_forward,
          "Broadcast addition of param over batch dimension",
          py::arg("x"), py::arg("param"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
