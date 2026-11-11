#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_dim1_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    cudaStream_t stream);

extern "C" void scalar_div_launcher(
    const float* input,
    float* output,
    int64_t numel,
    double divisor,
    cudaStream_t stream);

// Wrapper for log_softmax along dim=1
torch::Tensor log_softmax_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional");

    // Ensure contiguous layout as launcher expects
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto batch = x.size(0);
    auto channels = x.size(1);
    auto spatial_size = x.size(2) * x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        spatial_size,
        stream
    );

    return output;
}

// Wrapper for scalar division
torch::Tensor scalar_div(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous layout as launcher expects
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto output = torch::empty_like(x);
    auto numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    scalar_div_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        divisor,
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_dim1", &log_softmax_dim1,
          "Compute log_softmax along dimension 1",
          py::arg("x"));
    m.def("scalar_div", &scalar_div,
          "Divide tensor by scalar",
          py::arg("x"), py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
