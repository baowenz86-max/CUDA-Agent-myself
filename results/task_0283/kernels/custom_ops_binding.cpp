#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void elu_forward_launcher(
    const float* input,
    float* output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements,
    cudaStream_t stream);

extern "C" void tanh_forward_launcher(
    const float* input,
    float* output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements,
    cudaStream_t stream);

namespace {

void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");
}

torch::Tensor elu_forward(torch::Tensor x) {
    check_input(x);
    auto output = torch::empty_like(x);

    int dim0 = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int dim3 = x.size(3);
    int stride0 = x.stride(0);
    int stride1 = x.stride(1);
    int stride2 = x.stride(2);
    int stride3 = x.stride(3);
    int total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    elu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        total_elements,
        stream);

    return output;
}

torch::Tensor tanh_forward(torch::Tensor x) {
    check_input(x);
    auto output = torch::empty_like(x);

    int dim0 = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    int dim3 = x.size(3);
    int stride0 = x.stride(0);
    int stride1 = x.stride(1);
    int stride2 = x.stride(2);
    int stride3 = x.stride(3);
    int total_elements = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        total_elements,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "Apply ELU activation",
          pybind11::arg("x"));
    m.def("tanh_forward", &tanh_forward, "Apply tanh activation",
          pybind11::arg("x"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
