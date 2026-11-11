#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void instance_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
);

// PyTorch binding for instance_norm_forward
torch::Tensor instance_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps = 1e-05f
) {
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    // Contiguous inputs for launcher
    auto x_contig = x.contiguous();
    auto weight_contig = weight.defined() ? weight.contiguous() : torch::Tensor();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    if (weight_contig.defined()) {
        TORCH_CHECK(weight_contig.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight_contig.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight_contig.numel() == C, "weight must have C elements");
    }
    if (bias_contig.defined()) {
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_contig.numel() == C, "bias must have C elements");
    }

    // Allocate output (same shape as input)
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    instance_norm_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_contig.defined() ? weight_contig.data_ptr<float>() : nullptr,
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        eps,
        N,
        C,
        HW,
        stream
    );

    return y;
}

// Registration function
void register_instance_norm_forward(pybind11::module& m) {
    m.def(
        "instance_norm_forward",
        &instance_norm_forward,
        "Instance normalization forward",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("eps") = 1e-05f
    );
}

// Auto-register
REGISTER_BINDING(instance_norm_forward, register_instance_norm_forward);
