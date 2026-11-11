#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void constant_pad1d_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t seq,
    int64_t width,
    int64_t pad_left,
    int64_t pad_right,
    double value,
    cudaStream_t stream);

extern "C" void sum_dim2_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t seq,
    int64_t width,
    cudaStream_t stream);

torch::Tensor constant_pad1d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    double value) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, seq, width)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t width = x.size(2);
    int64_t width_out = width + pad_left + pad_right;

    auto output = torch::empty({batch, seq, width_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad1d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        seq,
        width,
        pad_left,
        pad_right,
        value,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad1d_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor sum_dim2_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, seq, width)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t width = x.size(2);

    auto output = torch::empty({batch, seq}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim2_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch,
        seq,
        width,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim2_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad1d_forward", &constant_pad1d_forward,
          "Constant padding along last dim",
          py::arg("x"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("value"));
    m.def("sum_dim2_forward", &sum_dim2_forward,
          "Sum along last dim",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
