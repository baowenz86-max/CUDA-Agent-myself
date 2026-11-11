#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream);

extern "C" void selu_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

// Linear forward wrapper
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Check device
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    // Check contiguity
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // Check shapes
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [out_features]");

    int N = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight.size(1) must match input.size(1)");
    TORCH_CHECK(bias.size(0) == out_features, "bias.size(0) must match out_features");

    // Allocate output
    auto output = torch::empty({N, out_features}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    linear_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        in_features,
        out_features,
        stream
    );

    return output;
}

// SELU forward wrapper
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    // SELU supports strides, but the launcher assumes contiguous; we make contiguous if necessary
    auto input_contig = input.contiguous();

    int total = input_contig.numel();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );

    // If input was non-contiguous, we need to return a tensor with the same shape but contiguous;
    // The contract says output shape same as input, but layout likely contiguous.
    // Since reference uses contiguous, this is fine.
    return output;
}

// Sigmoid forward wrapper
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto input_contig = input.contiguous();
    int total = input_contig.numel();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sigmoid_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward, "Linear forward: y = input @ weight^T + bias",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("selu_forward", &selu_forward, "SELU activation",
          py::arg("input"));
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid activation",
          py::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
