#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// External launcher declarations from custom_ops.cu
extern "C" void relu_add_div6_launcher(float* output,
                                        const float* input,
                                        int64_t numel,
                                        cudaStream_t stream);

extern "C" void hardswish_launcher(float* output,
                                    const float* input,
                                    int64_t numel,
                                    cudaStream_t stream);

namespace {

torch::Tensor relu_add_div6(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_add_div6: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_add_div6: input must be float32");
    TORCH_CHECK(x.defined(), "relu_add_div6: input tensor is undefined");

    // The kernel handles arbitrary strides via raw pointer and linear index, so no contiguity check is needed.
    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_add_div6_launcher(output.data_ptr<float>(),
                           x.data_ptr<float>(),
                           numel,
                           stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_add_div6: CUDA error: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor hardswish(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardswish: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardswish: input must be float32");
    TORCH_CHECK(x.defined(), "hardswish: input tensor is undefined");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardswish_launcher(output.data_ptr<float>(),
                       x.data_ptr<float>(),
                       numel,
                       stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardswish: CUDA error: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_add_div6", &relu_add_div6,
          "relu_add_div6(x) -> Tensor\n"
          "Equivalent to F.relu(x + 3) / 6.",
          py::arg("x"));
    m.def("hardswish", &hardswish,
          "hardswish(x) -> Tensor\n"
          "Hardswish activation: x * relu6(x + 3) / 6.",
          py::arg("x"));
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
