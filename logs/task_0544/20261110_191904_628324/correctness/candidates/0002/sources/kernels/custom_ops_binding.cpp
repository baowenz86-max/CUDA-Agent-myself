#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    float* output_ptr,
    const float* input_ptr,
    int64_t numel,
    int64_t input_offset,
    int64_t output_offset,
    cudaStream_t stream);

extern "C" void batch_norm_eval_launcher(
    float* output_ptr,
    const float* input_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    const float* running_mean_ptr,
    const float* running_var_ptr,
    float eps,
    int64_t outer_size,
    int64_t C,
    int64_t inner_size,
    int64_t in_stride_outer,
    int64_t in_stride_channel,
    int64_t in_stride_inner,
    int64_t out_stride_outer,
    int64_t out_stride_channel,
    int64_t out_stride_inner,
    cudaStream_t stream);

// ReLU wrapper
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    // Ensure contiguous for simplified launcher; reference is contiguous, so no harm.
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        0, // input_offset (contiguous index offset 0)
        0, // output_offset
        stream);
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_forward_launcher failed");
    return output;
}

// BatchNorm eval wrapper
torch::Tensor batch_norm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    TORCH_CHECK(input.is_cuda(), "batch_norm_eval_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda() && running_mean.is_cuda() && running_var.is_cuda(),
                "batch_norm_eval_forward: all parameter tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32 &&
                bias.dtype() == torch::kFloat32 && running_mean.dtype() == torch::kFloat32 &&
                running_var.dtype() == torch::kFloat32,
                "batch_norm_eval_forward: all tensors must be float32");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && running_mean.dim() == 1 && running_var.dim() == 1,
                "batch_norm_eval_forward: parameter tensors must be 1D");
    TORCH_CHECK(weight.size(0) == bias.size(0) && weight.size(0) == running_mean.size(0) &&
                weight.size(0) == running_var.size(0),
                "batch_norm_eval_forward: parameter tensor sizes must match");
    TORCH_CHECK(input.dim() >= 2 && input.size(1) == weight.size(0),
                "batch_norm_eval_forward: input channel dimension must match parameter size");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto running_mean_contig = running_mean.contiguous();
    auto running_var_contig = running_var.contiguous();

    auto output = torch::empty_like(input_contig);

    int64_t outer_size = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t inner_size = input_contig.numel() / (outer_size * C);

    // After contiguous, strides are in elements
    int64_t in_stride_outer = C * inner_size;
    int64_t in_stride_channel = inner_size;
    int64_t in_stride_inner = 1;
    int64_t out_stride_outer = C * inner_size;
    int64_t out_stride_channel = inner_size;
    int64_t out_stride_inner = 1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm_eval_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        static_cast<float>(eps),
        outer_size,
        C,
        inner_size,
        in_stride_outer,
        in_stride_channel,
        in_stride_inner,
        out_stride_outer,
        out_stride_channel,
        out_stride_inner,
        stream);
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "batch_norm_eval_launcher failed");
    return output;
}

void register_custom_ops(py::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward kernel",
          py::arg("input"));
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward, "BatchNorm eval forward kernel",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
