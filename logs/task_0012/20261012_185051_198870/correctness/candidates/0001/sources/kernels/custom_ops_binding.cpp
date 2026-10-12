// Binding for custom CUDA ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare raw launchers from custom_ops.cu
extern "C" void relu_sqrt_forward(
    const float* input, float* output, int numel, cudaStream_t stream);
extern "C" void silu_forward(
    const float* input, float* output, int numel, cudaStream_t stream);
extern "C" void mha_forward(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output, int seq_len, int batch, int embed_dim, int num_heads,
    cudaStream_t stream);

static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor relu_sqrt_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input);
    relu_sqrt_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        get_stream()
    );
    return output;
}

torch::Tensor silu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input);
    silu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        get_stream()
    );
    return output;
}

torch::Tensor mha_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t num_heads) {
    // Validate tensors
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

    // Check shapes
    TORCH_CHECK(query.dim() == 3, "query must be 3D (seq_len, batch, embed_dim)");
    TORCH_CHECK(key.dim() == 3, "key must be 3D");
    TORCH_CHECK(value.dim() == 3, "value must be 3D");
    int64_t seq_len = query.size(0);
    int64_t batch = query.size(1);
    int64_t embed_dim = query.size(2);
    TORCH_CHECK(key.size(0) == seq_len && key.size(1) == batch && key.size(2) == embed_dim,
                "key shape mismatch");
    TORCH_CHECK(value.size(0) == seq_len && value.size(1) == batch && value.size(2) == embed_dim,
                "value shape mismatch");

    TORCH_CHECK(in_proj_weight.dim() == 2 && in_proj_weight.size(0) == 3 * embed_dim &&
                in_proj_weight.size(1) == embed_dim, "in_proj_weight shape must be (3*embed_dim, embed_dim)");
    TORCH_CHECK(in_proj_bias.dim() == 1 && in_proj_bias.size(0) == 3 * embed_dim,
                "in_proj_bias shape must be (3*embed_dim)");
    TORCH_CHECK(out_proj_weight.dim() == 2 && out_proj_weight.size(0) == embed_dim &&
                out_proj_weight.size(1) == embed_dim, "out_proj_weight shape must be (embed_dim, embed_dim)");
    TORCH_CHECK(out_proj_bias.dim() == 1 && out_proj_bias.size(0) == embed_dim,
                "out_proj_bias shape must be (embed_dim)");
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

    auto output = torch::empty_like(query);
    mha_forward(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(seq_len),
        static_cast<int>(batch),
        static_cast<int>(embed_dim),
        static_cast<int>(num_heads),
        get_stream()
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_sqrt_forward", &relu_sqrt_forward, "relu + sqrt forward",
          pybind11::arg("input"));
    m.def("silu_forward", &silu_forward, "silu forward",
          pybind11::arg("input"));
    m.def("mha_forward", &mha_forward, "multi-head attention forward",
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("in_proj_weight"),
          pybind11::arg("in_proj_bias"),
          pybind11::arg("out_proj_weight"),
          pybind11::arg("out_proj_bias"),
          pybind11::arg("num_heads"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
