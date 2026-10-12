// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void relu_sqrt_forward_launcher(
    float* output, const float* input, int64_t numel, cudaStream_t stream);
void silu_forward_launcher(
    float* output, const float* input, int64_t numel, cudaStream_t stream);
void in_proj_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int seq_len, int batch, int embed_dim, cudaStream_t stream);
void mha_forward_launcher(
    const float* q, const float* k, const float* v,
    float* output, float* attn_weights,
    int batch, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream);
void out_proj_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int seq_len, int batch, int embed_dim, cudaStream_t stream);
}

// Wrapper for fused ReLU + sqrt
torch::Tensor relu_sqrt_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_sqrt_forward_launcher(
        output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

// Wrapper for SiLU
torch::Tensor silu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_forward_launcher(
        output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

// Wrapper for multi-head attention (including in-projection and out-projection)
std::tuple<torch::Tensor, torch::Tensor> mha_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t num_heads)
{
    // Basic checks
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be CUDA");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be CUDA");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be CUDA");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(in_proj_weight.is_contiguous(), "in_proj_weight must be contiguous");
    TORCH_CHECK(in_proj_bias.is_contiguous(), "in_proj_bias must be contiguous");
    TORCH_CHECK(out_proj_weight.is_contiguous(), "out_proj_weight must be contiguous");
    TORCH_CHECK(out_proj_bias.is_contiguous(), "out_proj_bias must be contiguous");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(in_proj_bias.dtype() == torch::kFloat32, "in_proj_bias must be float32");
    TORCH_CHECK(out_proj_weight.dtype() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(out_proj_bias.dtype() == torch::kFloat32, "out_proj_bias must be float32");

    // Derive dimensions from the input tensors
    int64_t seq_len = query.size(0);
    int64_t batch = query.size(1);
    int64_t embed_dim = query.size(2);
    int64_t head_dim = embed_dim / num_heads;
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
    TORCH_CHECK(key.size(0) == seq_len && key.size(1) == batch && key.size(2) == embed_dim, "key shape mismatch");
    TORCH_CHECK(value.size(0) == seq_len && value.size(1) == batch && value.size(2) == embed_dim, "value shape mismatch");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim, "in_proj_weight shape mismatch");
    TORCH_CHECK(in_proj_bias.numel() == 3 * embed_dim, "in_proj_bias size mismatch");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim, "out_proj_weight shape mismatch");
    TORCH_CHECK(out_proj_bias.numel() == embed_dim, "out_proj_bias size mismatch");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Step 1: Compute in-projection: output shape (seq_len, batch, 3*embed_dim)
    auto in_proj_output = torch::empty({seq_len, batch, 3 * embed_dim}, options);
    in_proj_forward_launcher(
        query.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        in_proj_output.data_ptr<float>(),
        seq_len, batch, embed_dim, stream);

    // Step 2: Create contiguous copies of q, k, v slices.
    // The attention kernel assumes contiguous memory with layout (seq_len, batch, embed_dim).
    auto q = in_proj_output.narrow(2, 0, embed_dim).contiguous();
    auto k = in_proj_output.narrow(2, embed_dim, embed_dim).contiguous();
    auto v = in_proj_output.narrow(2, 2 * embed_dim, embed_dim).contiguous();

    // Step 3: Allocate attention output (seq_len, batch, embed_dim)
    auto attn_output = torch::empty({seq_len, batch, embed_dim}, options);

    // Step 4: Allocate attention weights (batch, num_heads, seq_len, seq_len)
    auto attn_weights = torch::empty({batch, num_heads, seq_len, seq_len}, options);

    // Step 5: Call attention kernel
    mha_forward_launcher(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        batch, seq_len, num_heads, head_dim, stream);

    // Step 6: Compute out-projection: (seq_len, batch, embed_dim) -> (seq_len, batch, embed_dim)
    auto final_output = torch::empty({seq_len, batch, embed_dim}, options);
    out_proj_forward_launcher(
        attn_output.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        final_output.data_ptr<float>(),
        seq_len, batch, embed_dim, stream);

    return std::make_tuple(final_output, attn_weights);
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_sqrt_forward", &relu_sqrt_forward, "Fused ReLU + sqrt forward",
          pybind11::arg("input"));
    m.def("silu_forward", &silu_forward, "SiLU forward",
          pybind11::arg("input"));
    m.def("mha_forward", &mha_forward, "Multi-head attention forward (returns output and weights)",
          pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("in_proj_weight"), pybind11::arg("in_proj_bias"),
          pybind11::arg("out_proj_weight"), pybind11::arg("out_proj_bias"),
          pybind11::arg("num_heads"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
