#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    double min_val,
    double max_val,
    cudaStream_t stream);

extern "C" void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t rows,
    int64_t F,
    double eps,
    int64_t row_stride_in,
    int64_t row_stride_out,
    cudaStream_t stream);

// clamp_forward wrapper
torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "clamp_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clamp_forward: input must be float32");

    // Ensure contiguous for launcher assumptions (though launcher supports linear indexing)
    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        min_val,
        max_val,
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// layer_norm_forward wrapper
torch::Tensor layer_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "layer_norm_forward: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "layer_norm_forward: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "layer_norm_forward: bias must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "layer_norm_forward: x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "layer_norm_forward: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "layer_norm_forward: bias must be float32");

    TORCH_CHECK(x.dim() >= 1, "layer_norm_forward: x must have at least 1 dimension");
    const int64_t F = x.size(-1);
    TORCH_CHECK(weight.numel() == F, "layer_norm_forward: weight size must match last dimension of x");
    TORCH_CHECK(bias.numel() == F, "layer_norm_forward: bias size must match last dimension of x");

    // Ensure all tensors are contiguous (launcher assumes last dim stride 1)
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    const int64_t rows = x_contig.numel() / F;

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    layer_norm_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        rows,
        F,
        eps,
        F, // row_stride_in (contiguous)
        F, // row_stride_out (contiguous)
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "layer_norm_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward,
          "Clamp forward",
          py::arg("x"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);

    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
