#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the launcher from fused_attention.cu
extern "C" void fused_attention_forward_launcher(
    const float* x, const float* y,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output,
    int B, int S_orig, int repeat_times,
    int E, int H,
    float scaling_factor,
    cudaStream_t stream
);

torch::Tensor fused_attention_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t embed_dim,
    int64_t num_heads,
    int64_t repeat_interleave_times,
    double scaling_factor
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be a CUDA tensor");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be a CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be a CUDA tensor");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(in_proj_weight.is_contiguous(), "in_proj_weight must be contiguous");
    TORCH_CHECK(in_proj_bias.is_contiguous(), "in_proj_bias must be contiguous");
    TORCH_CHECK(out_proj_weight.is_contiguous(), "out_proj_weight must be contiguous");
    TORCH_CHECK(out_proj_bias.is_contiguous(), "out_proj_bias must be contiguous");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(in_proj_bias.dtype() == torch::kFloat32, "in_proj_bias must be float32");
    TORCH_CHECK(out_proj_weight.dtype() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(out_proj_bias.dtype() == torch::kFloat32, "out_proj_bias must be float32");

    // Check dimensions
    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, S, E]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D [B, S, E]");
    int64_t B = x.size(0);
    int64_t S_orig = x.size(1);
    int64_t E = x.size(2);
    TORCH_CHECK(y.size(0) == B && y.size(1) == S_orig && y.size(2) == E, "y shape mismatch");
    TORCH_CHECK(E == embed_dim, "embed_dim mismatch");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim, "in_proj_weight shape must be [3E, E]");
    TORCH_CHECK(in_proj_bias.numel() == 3 * embed_dim, "in_proj_bias must have 3E elements");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim, "out_proj_weight shape must be [E, E]");
    TORCH_CHECK(out_proj_bias.numel() == embed_dim, "out_proj_bias must have E elements");
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

    // Compute output size: S_rep = S_orig * repeat_interleave_times
    int64_t S_rep = S_orig * repeat_interleave_times;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor output = torch::empty({B, S_rep, E}, options);

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    fused_attention_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(S_orig),
        static_cast<int>(repeat_interleave_times),
        static_cast<int>(E),
        static_cast<int>(num_heads),
        static_cast<float>(scaling_factor),
        stream
    );

    return output;
}

void register_fused_attention(pybind11::module& m) {
    m.def(
        "fused_attention_forward",
        &fused_attention_forward,
        "Fused repeat interleave + multihead attention + tan scaling",
        pybind11::arg("x"),
        pybind11::arg("y"),
        pybind11::arg("in_proj_weight"),
        pybind11::arg("in_proj_bias"),
        pybind11::arg("out_proj_weight"),
        pybind11::arg("out_proj_bias"),
        pybind11::arg("embed_dim"),
        pybind11::arg("num_heads"),
        pybind11::arg("repeat_interleave_times"),
        pybind11::arg("scaling_factor")
    );
}

REGISTER_BINDING(fused_attention, register_fused_attention);