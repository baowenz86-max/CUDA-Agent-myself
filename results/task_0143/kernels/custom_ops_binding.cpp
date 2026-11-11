#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(const float* input,
                                       float* output,
                                       int64_t n,
                                       cudaStream_t stream);

extern "C" void clamp_hardswish_forward_launcher(const float* input,
                                                 float* output,
                                                 int64_t n,
                                                 cudaStream_t stream);

// Helper to validate and get contiguous float32 CUDA tensor pointer
static inline const float* check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    return x.data_ptr<float>();
}

static inline torch::Tensor allocate_output_like(const torch::Tensor& x) {
    return torch::empty_like(x);
}

static inline cudaStream_t current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper for clamp_forward
torch::Tensor clamp_forward(torch::Tensor x) {
    const float* input = check_input(x);
    auto output = allocate_output_like(x);
    float* out_ptr = output.data_ptr<float>();
    int64_t n = x.numel();
    cudaStream_t stream = current_stream();
    clamp_forward_launcher(input, out_ptr, n, stream);
    return output;
}

// Wrapper for clamp_hardswish_forward
torch::Tensor clamp_hardswish_forward(torch::Tensor x) {
    const float* input = check_input(x);
    auto output = allocate_output_like(x);
    float* out_ptr = output.data_ptr<float>();
    int64_t n = x.numel();
    cudaStream_t stream = current_stream();
    clamp_hardswish_forward_launcher(input, out_ptr, n, stream);
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward, "Clamp tensor to [0,1]",
          pybind11::arg("x"));
    m.def("clamp_hardswish_forward", &clamp_hardswish_forward,
          "Fused clamp and hardswish",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
