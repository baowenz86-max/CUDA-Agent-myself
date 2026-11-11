#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Extern declarations for launchers in custom_ops.cu
extern "C" void custom_group_norm_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int B,
    int C,
    int H,
    int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

extern "C" void custom_permute_sum_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream
);

// Custom GroupNorm wrapper
torch::Tensor custom_group_norm(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    int64_t num_groups,
    int64_t num_channels,
    double eps
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, C, H, W)");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(num_channels % num_groups == 0, "num_channels must be divisible by num_groups");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(C == num_channels, "x.size(1) must match num_channels");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight->scalar_type() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->numel() == C, "weight must have numel equal to num_channels");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->numel() == C, "bias must have numel equal to num_channels");
        bias_ptr = bias->data_ptr<float>();
    }

    // Allocate output of same shape as input
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_group_norm_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        B, C, H, W,
        static_cast<int>(num_groups),
        static_cast<float>(eps),
        stream
    );

    return y;
}

// Custom PermuteSum wrapper: x [B,C,H,W] -> [B,H,W,1]
torch::Tensor custom_permute_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, C, H, W)");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape [B,H,W,1]
    auto out = torch::empty({B, H, W, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_permute_sum_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_group_norm", &custom_group_norm,
          "Custom GroupNorm forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("num_groups"), py::arg("num_channels"), py::arg("eps"));
    m.def("custom_permute_sum", &custom_permute_sum,
          "Custom permute and sum over channel dimension");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
