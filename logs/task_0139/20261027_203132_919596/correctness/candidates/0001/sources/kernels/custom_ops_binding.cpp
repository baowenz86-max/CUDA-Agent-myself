#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream);

extern "C" void selu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

namespace {

// Helper: returns a contiguous float32 CUDA tensor copy if input is not suitable,
// otherwise returns the input tensor itself (increments reference count).
torch::Tensor ensure_contiguous_float32_cuda(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    if (input.is_contiguous()) {
        return input;
    }
    return input.contiguous();
}

}  // namespace

// Python-callable wrapper for leaky_relu_forward
torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope = 0.01) {
    auto x = ensure_contiguous_float32_cuda(input);
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        static_cast<float>(negative_slope),
        stream);

    // If input was non-contiguous, output has same shape but contiguous layout;
    // that's acceptable since reference returns contiguous tensor with same shape.
    return output;
}

// Python-callable wrapper for selu_forward
torch::Tensor selu_forward(torch::Tensor input) {
    auto x = ensure_contiguous_float32_cuda(input);
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          pybind11::arg("input"),
          pybind11::arg("negative_slope") = 0.01);
    m.def("selu_forward", &selu_forward,
          "SELU forward",
          pybind11::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
