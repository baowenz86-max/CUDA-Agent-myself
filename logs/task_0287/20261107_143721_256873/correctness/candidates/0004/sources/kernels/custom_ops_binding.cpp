// PyTorch binding for custom ops defined in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int spatial_size,
    long long input_stride_n,
    long long input_stride_c,
    long long output_stride_n,
    long long output_stride_c,
    cudaStream_t stream);

extern "C" void tanh_forward_launcher(
    float* output,
    const float* input,
    long long total,
    cudaStream_t stream);

// BatchNorm2d eval forward
// Inputs: x (N,C,H,W) float32 CUDA, weight (C) float32 CUDA optional, bias (C) float32 CUDA optional,
//         running_mean (C) float32 CUDA, running_var (C) float32 CUDA, eps float
torch::Tensor batch_norm_eval_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int spatial_size = H * W;

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean must have C elements");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.numel() == C, "running_var must have C elements");

    const float* weight_ptr = nullptr;
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.numel() == C, "weight must have C elements");
        weight_ptr = weight.data_ptr<float>();
    }

    const float* bias_ptr = nullptr;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C, "bias must have C elements");
        bias_ptr = bias.data_ptr<float>();
    }

    // Handle non-contiguous input by making it contiguous
    torch::Tensor x_contig = x.contiguous();
    // Output has same shape as (possibly non-contiguous) input? Contract says shape same,
    // but if x is non-contiguous, we produce contiguous output with same logical shape.
    // Reference output is contiguous (stride [393216, 131072, 256, 1] for standard contiguous).
    auto output = torch::empty_like(x_contig);

    long long input_stride_n = x_contig.stride(0);
    long long input_stride_c = x_contig.stride(1);
    long long output_stride_n = output.stride(0);
    long long output_stride_c = output.stride(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_eval_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        C,
        spatial_size,
        input_stride_n,
        input_stride_c,
        output_stride_n,
        output_stride_c,
        stream);

    return output;
}

// Tanh forward
// Input: x float32 CUDA tensor, arbitrary strides
torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    torch::Tensor x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    long long total = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        total,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm2d eval forward",
          py::arg("x"),
          py::arg("weight").none(true),
          py::arg("bias").none(true),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05);

    m.def("tanh_forward", &tanh_forward,
          "Tanh activation forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);