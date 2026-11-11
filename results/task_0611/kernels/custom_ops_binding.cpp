#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void layer_norm_sum_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int B,
    int S,
    int F,
    int stride_b,
    int stride_s,
    int stride_f,
    int out_stride_b,
    int out_stride_f,
    double eps,
    cudaStream_t stream);

// Wrapper for layer_norm_sum operator
static torch::Tensor layer_norm_sum_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps = 1e-5) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, S, F]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t F = x.size(2);

    TORCH_CHECK(weight.size(0) == F, "weight size must match x last dimension");
    TORCH_CHECK(bias.size(0) == F, "bias size must match x last dimension");

    // Create output tensor of shape [B, F]
    auto out = torch::empty({B, F}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with explicit strides
    layer_norm_sum_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(F),
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(2)),
        static_cast<int>(out.stride(0)),
        static_cast<int>(out.stride(1)),
        eps,
        stream);

    return out;
}

// Registration function
static void register_layer_norm_sum(pybind11::module& m) {
    m.def("layer_norm_sum",
          &layer_norm_sum_forward,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-05);
}

// Auto-register
REGISTER_BINDING(layer_norm_sum, register_layer_norm_sum);
