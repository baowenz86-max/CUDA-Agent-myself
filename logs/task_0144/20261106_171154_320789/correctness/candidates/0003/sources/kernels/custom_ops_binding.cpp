#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void batch_norm_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_l,
    cudaStream_t stream
);

torch::Tensor batch_norm_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps = 1e-5f
) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "batch_norm_inference: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "batch_norm_inference: x must be float32");
    TORCH_CHECK(x.dim() == 3, "batch_norm_inference: x must be 3D (N, C, L)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    // Validate affine parameters
    TORCH_CHECK(weight.is_cuda(), "batch_norm_inference: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "batch_norm_inference: weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "batch_norm_inference: weight must be contiguous");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "batch_norm_inference: weight must have shape [C]");

    TORCH_CHECK(bias.is_cuda(), "batch_norm_inference: bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "batch_norm_inference: bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "batch_norm_inference: bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "batch_norm_inference: bias must have shape [C]");

    TORCH_CHECK(running_mean.is_cuda(), "batch_norm_inference: running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "batch_norm_inference: running_mean must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "batch_norm_inference: running_mean must be contiguous");
    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == C, "batch_norm_inference: running_mean must have shape [C]");

    TORCH_CHECK(running_var.is_cuda(), "batch_norm_inference: running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "batch_norm_inference: running_var must be float32");
    TORCH_CHECK(running_var.is_contiguous(), "batch_norm_inference: running_var must be contiguous");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == C, "batch_norm_inference: running_var must have shape [C]");

    // Handle non-contiguous input by making a contiguous copy if needed.
    torch::Tensor x_contig = x.is_contiguous() ? x : x.contiguous();

    // Allocate output with same shape and strides as original x (or contiguous equivalent)
    torch::Tensor y = torch::empty_like(x);
    // Ensure y is contiguous for the launcher's linear indexing assumption.
    torch::Tensor y_contig = y.is_contiguous() ? y : y.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Calculate strides for x_contig (contiguous)
    int64_t stride_n = C * L;
    int64_t stride_c = L;
    int64_t stride_l = 1;

    // Call launcher with raw pointers
    batch_norm_inference_launcher(
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N,
        C,
        L,
        stride_n,
        stride_c,
        stride_l,
        stream
    );

    // If y was originally non-contiguous, copy back to original layout.
    if (!y.is_contiguous()) {
        y.copy_(y_contig);
    }

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def(
        "batch_norm_inference",
        &batch_norm_inference,
        "BatchNorm1d inference forward",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("running_mean"),
        pybind11::arg("running_var"),
        pybind11::arg("eps") = 1e-5f
    );
}

REGISTER_BINDING(custom_ops, register_custom_ops);
