// Binding for linear_forward: y = x @ W^T + bias
// Only float32, contiguous, CUDA tensors supported.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from linear_kernel.cu
extern "C" void linear_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size,
    int seq_len,
    int in_features,
    int out_features,
    cudaStream_t stream
);

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, seq_len, in_features)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (out_features, in_features)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_features)");
    TORCH_CHECK(input.size(-1) == weight.size(1), "input feature dim mismatch");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight/bias out_features mismatch");

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int in_features = input.size(2);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, seq_len, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size, seq_len, in_features, out_features,
        stream
    );

    return output;
}

void register_linear(py::module& m) {
    m.def("linear_forward", &linear_forward, "Linear forward (y = x @ W^T + b)",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(linear, register_linear);
