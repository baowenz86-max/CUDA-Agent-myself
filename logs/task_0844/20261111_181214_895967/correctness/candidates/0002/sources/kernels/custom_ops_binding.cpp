// PyTorch binding for custom attention and mean operations.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    const float* attn_mask,
    int L, int N, int E, int num_heads,
    float* output, cudaStream_t stream);

extern "C" void mean_forward_launcher(
    const float* input,
    int L, int N, int E, int dim,
    float* output, cudaStream_t stream);

torch::Tensor check_contiguous_float32_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    return t.contiguous();
}

torch::Tensor multihead_attention_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor in_proj_weight, torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias,
    c10::optional<torch::Tensor> attn_mask,
    int64_t embed_dim, int64_t num_heads) {

    query = check_contiguous_float32_cuda(query, "query");
    key = check_contiguous_float32_cuda(key, "key");
    value = check_contiguous_float32_cuda(value, "value");
    in_proj_weight = check_contiguous_float32_cuda(in_proj_weight, "in_proj_weight");
    in_proj_bias = check_contiguous_float32_cuda(in_proj_bias, "in_proj_bias");
    out_proj_weight = check_contiguous_float32_cuda(out_proj_weight, "out_proj_weight");
    out_proj_bias = check_contiguous_float32_cuda(out_proj_bias, "out_proj_bias");

    const float* attn_mask_ptr = nullptr;
    if (attn_mask.has_value() && attn_mask->defined()) {
        attn_mask = check_contiguous_float32_cuda(*attn_mask, "attn_mask");
        attn_mask_ptr = attn_mask->data_ptr<float>();
    }

    TORCH_CHECK(query.dim() == 3, "query must be 3D [L, N, E]");
    int L = query.size(0);
    int N = query.size(1);
    int E = query.size(2);
    TORCH_CHECK(key.sizes() == query.sizes(), "key must have same shape as query");
    TORCH_CHECK(value.sizes() == query.sizes(), "value must have same shape as query");
    TORCH_CHECK(in_proj_weight.dim() == 2);
    TORCH_CHECK(in_proj_weight.size(0) == 3 * E);
    TORCH_CHECK(in_proj_weight.size(1) == E);
    TORCH_CHECK(in_proj_bias.dim() == 1 && in_proj_bias.size(0) == 3 * E);
    TORCH_CHECK(out_proj_weight.dim() == 2 && out_proj_weight.size(0) == E && out_proj_weight.size(1) == E);
    TORCH_CHECK(out_proj_bias.dim() == 1 && out_proj_bias.size(0) == E);
    if (attn_mask_ptr) {
        TORCH_CHECK(attn_mask->dim() == 2 && attn_mask->size(0) == L && attn_mask->size(1) == L);
    }

    TORCH_CHECK(embed_dim == E, "embed_dim mismatch");
    TORCH_CHECK(num_heads > 0 && E % num_heads == 0, "num_heads must divide embed_dim");

    auto output = torch::empty_like(query);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    multihead_attention_forward_launcher(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(), in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(), out_proj_bias.data_ptr<float>(),
        attn_mask_ptr,
        L, N, E, static_cast<int>(num_heads),
        output.data_ptr<float>(), stream);

    return output;
}

torch::Tensor mean_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    input = check_contiguous_float32_cuda(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [L, N, E]");

    int L = input.size(0);
    int N = input.size(1);
    int E = input.size(2);

    auto output = torch::empty({L, E}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_forward_launcher(
        input.data_ptr<float>(),
        L, N, E, static_cast<int>(dim),
        output.data_ptr<float>(), stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multihead attention forward (custom CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("attn_mask"), py::arg("embed_dim"), py::arg("num_heads"));
    m.def("mean_forward", &mean_forward,
          "Mean over batch dimension (dim=1)",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
