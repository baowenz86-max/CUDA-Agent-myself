#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare CUDA launcher from custom_ops.cu
extern "C" void batch_norm1d_eval_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    int N,
    int C,
    int L,
    float eps,
    cudaStream_t stream
);

namespace {

torch::Tensor batch_norm1d_eval_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.scalar_type() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.scalar_type() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");
    TORCH_CHECK(running_mean.numel() == running_var.numel(), "running_mean and running_var must have same size");

    int N = input.size(0);
    int C = input.size(1);
    int L = input.size(2);

    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channel dimension");

    // Optional tensors
    torch::Tensor weight_t = weight.has_value() ? weight.value() : torch::Tensor();
    torch::Tensor bias_t = bias.has_value() ? bias.value() : torch::Tensor();

    if (weight_t.defined()) {
        TORCH_CHECK(weight_t.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight_t.scalar_type() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight_t.dim() == 1 && weight_t.numel() == C, "weight must have shape [C]");
    }
    if (bias_t.defined()) {
        TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_t.scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.numel() == C, "bias must have shape [C]");
    }

    // Ensure contiguity
    auto input_contig = input.contiguous();
    auto running_mean_contig = running_mean.contiguous();
    auto running_var_contig = running_var.contiguous();
    auto weight_contig = weight_t.defined() ? weight_t.contiguous() : torch::Tensor();
    auto bias_contig = bias_t.defined() ? bias_t.contiguous() : torch::Tensor();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm1d_eval_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_contig.defined() ? weight_contig.data_ptr<float>() : nullptr,
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        N,
        C,
        L,
        static_cast<float>(eps),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

void register_batch_norm1d_eval_forward(pybind11::module& m) {
    m.def(
        "batch_norm1d_eval_forward",
        &batch_norm1d_eval_forward,
        "BatchNorm1d evaluation forward (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("running_mean"),
        pybind11::arg("running_var"),
        pybind11::arg("eps")
    );
}

}  // namespace

REGISTER_BINDING(batch_norm1d_eval_forward, register_batch_norm1d_eval_forward);
