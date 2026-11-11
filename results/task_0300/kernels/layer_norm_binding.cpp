// Binding for LayerNorm forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int num_rows,
    int normalized_shape,
    float eps,
    cudaStream_t stream);

static torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t normalized_shape_int,
    double eps) {

    // Basic checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int64_t normalized_shape = normalized_shape_int;
    TORCH_CHECK(input.size(-1) == normalized_shape,
                "input last dimension must equal normalized_shape, got ",
                input.size(-1), " vs ", normalized_shape);
    TORCH_CHECK(weight.numel() == normalized_shape,
                "weight must have shape [normalized_shape]");
    TORCH_CHECK(bias.numel() == normalized_shape,
                "bias must have shape [normalized_shape]");

    // Make contiguous (safe for non-contiguous strides)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Allocate output same shape as input
    auto output = torch::empty_like(input_contig);

    int64_t num_rows = input_contig.numel() / normalized_shape;
    TORCH_CHECK(num_rows <= INT_MAX, "num_rows exceeds int range");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    layer_norm_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        static_cast<int>(num_rows),
        static_cast<int>(normalized_shape),
        static_cast<float>(eps),
        stream);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in layer_norm_forward: ", cudaGetErrorString(err));

    return output;
}

static void register_layer_norm(pybind11::module& m) {
    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("normalized_shape"),
          pybind11::arg("eps"));
}

REGISTER_BINDING(layer_norm, register_layer_norm);
