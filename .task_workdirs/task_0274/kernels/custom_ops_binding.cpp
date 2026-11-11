#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream);
extern "C" void abs_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream);

static torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "relu_forward: input must be float32");
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    long long numel = x_contig.numel();
    if (numel == 0) {
        return y;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream
    );
    return y;
}

static torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "abs_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "abs_forward: input must be float32");
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    long long numel = x_contig.numel();
    if (numel == 0) {
        return y;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream
    );
    return y;
}

static void register_custom_ops_binding(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward (CUDA)",
          py::arg("x"));
    m.def("abs_forward", &abs_forward, "Abs forward (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
