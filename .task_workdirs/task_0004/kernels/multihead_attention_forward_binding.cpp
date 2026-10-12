#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void multihead_attention_launcher(
    const float* x,
    const float* y,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* output,
    int B, int S, int E, int H,
    float dropout_prob,
    cudaStream_t stream
);

torch::Tensor multihead_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t embed_dim,
    int64_t num_heads,
    bool batch_first,
    double dropout)
{
    // Input validation
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be a CUDA tensor");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be a CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be a CUDA tensor");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be a CUDA tensor");

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

    TORCH_CHECK(batch_first, "only batch_first=True is supported");
    TORCH_CHECK(query.dim() == 3, "query must be 3D (B, S, E)");
    TORCH_CHECK(key.dim() == 3, "key must be 3D");
    TORCH_CHECK(value.dim() == 3, "value must be 3D");

    auto B = query.size(0);
    auto S = query.size(1);
    auto E = query.size(2);

    TORCH_CHECK(key.size(0) == B && key.size(1) == S && key.size(2) == E, "key shape mismatch");
    TORCH_CHECK(value.size(0) == B && value.size(1) == S && value.size(2) == E, "value shape mismatch");
    TORCH_CHECK(E == embed_dim, "embedding dimension mismatch");
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim, "in_proj_weight shape must be [3*E, E]");
    TORCH_CHECK(in_proj_bias.numel() == 3 * embed_dim, "in_proj_bias size must be 3*E");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim, "out_proj_weight shape must be [E, E]");
    TORCH_CHECK(out_proj_bias.numel() == embed_dim, "out_proj_bias size must be E");

    // Allocate output tensor
    auto output = torch::empty({B, S, E}, query.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    multihead_attention_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(E),
        static_cast<int>(num_heads),
        static_cast<float>(dropout),
        stream
    );

    return output;
}

void register_multihead_attention_forward(pybind11::module& m) {
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multihead attention forward (batch_first=True)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("in_proj_weight"),
          py::arg("in_proj_bias"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"),
          py::arg("embed_dim"),
          py::arg("num_heads"),
          py::arg("batch_first"),
          py::arg("dropout"));
}

REGISTER_BINDING(multihead_attention_forward, register_multihead_attention_forward);
