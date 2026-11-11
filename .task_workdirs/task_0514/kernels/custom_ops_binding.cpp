#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void layer_norm_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int S, int H, double eps,
    cudaStream_t stream);

extern "C" void mean_dim_forward_launcher(
    const float* input,
    float* output,
    int B, int S, int H,
    cudaStream_t stream);

// LayerNorm forward wrapper
torch::Tensor layer_norm_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps) {

    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, S, H)");

    int B = input.size(0);
    int S = input.size(1);
    int H = input.size(2);

    // Validate weight/bias if present
    if (weight.has_value()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->numel() == H, "weight size must match last dimension");
        TORCH_CHECK(weight->is_contiguous(), "weight must be contiguous");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->numel() == H, "bias size must match last dimension");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }

    // Ensure input is contiguous, as launcher assumes contiguous strides
    torch::Tensor input_contig = input.contiguous();

    // Allocate output with same shape
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    layer_norm_forward_launcher(
        input_contig.data_ptr<float>(),
        weight.has_value() ? weight->data_ptr<float>() : nullptr,
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, S, H, eps,
        stream);

    return output;
}

// Mean over dimension 1 wrapper
torch::Tensor mean_dim_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, S, H)");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int B = input.size(0);
    int S = input.size(1);
    int H = input.size(2);

    // Ensure input is contiguous
    torch::Tensor input_contig = input.contiguous();

    // Allocate output [B, H]
    auto output = torch::empty({B, H}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mean_dim_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, S, H,
        stream);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"));
    m.def("mean_dim_forward", &mean_dim_forward,
          "Mean over dim=1 (CUDA)",
          py::arg("input"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
