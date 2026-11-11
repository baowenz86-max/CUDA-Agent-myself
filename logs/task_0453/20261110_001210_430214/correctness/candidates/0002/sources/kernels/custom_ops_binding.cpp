#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" bool custom_equal_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream);

extern "C" float custom_l1_loss_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream);

static std::pair<const float*, const float*> check_and_get_ptrs(
    const torch::Tensor& x,
    const torch::Tensor& y,
    long long& numel,
    int& ndim)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();

    numel = x_contig.numel();
    ndim = x_contig.dim();

    return {x_contig.data_ptr<float>(), y_contig.data_ptr<float>()};
}

bool custom_equal(torch::Tensor x, torch::Tensor y) {
    long long numel = 0;
    int ndim = 0;
    auto [x_ptr, y_ptr] = check_and_get_ptrs(x, y, numel, ndim);

    const long long* x_strides = nullptr;
    const long long* y_strides = nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bool result = custom_equal_launcher(
        x_ptr,
        y_ptr,
        numel,
        x_strides,
        y_strides,
        ndim,
        stream);

    return result;
}

torch::Tensor custom_l1_loss(torch::Tensor x, torch::Tensor y) {
    long long numel = 0;
    int ndim = 0;
    auto [x_ptr, y_ptr] = check_and_get_ptrs(x, y, numel, ndim);

    const long long* x_strides = nullptr;
    const long long* y_strides = nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float mean = custom_l1_loss_launcher(
        x_ptr,
        y_ptr,
        numel,
        x_strides,
        y_strides,
        ndim,
        stream);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, x.get_device());
    torch::Tensor output = torch::empty({}, options);
    output.fill_(mean);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_equal", &custom_equal, "CUDA implementation of torch.equal",
          py::arg("x"), py::arg("y"));
    m.def("custom_l1_loss", &custom_l1_loss, "CUDA implementation of F.l1_loss with mean reduction",
          py::arg("x"), py::arg("y"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
