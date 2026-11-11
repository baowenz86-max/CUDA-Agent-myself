// custom_ops_binding.cpp: PyTorch bindings for custom CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_multihead_attention_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* avg_weights,
    int seq_len, int batch, int embed_dim, int num_heads,
    bool need_weights, bool average_attn_weights,
    cudaStream_t stream);

extern "C" void custom_cat_launcher(
    const float* t1, const float* t2, float* output,
    int outer_size, int inner_size, cudaStream_t stream);

extern "C" void custom_glu_launcher(
    const float* x, float* output,
    int outer_size, int inner_size, cudaStream_t stream);

extern "C" void custom_linear_launcher(
    const float* x, const float* weight, const float* bias, float* output,
    int outer_size, int in_features, int out_features, cudaStream_t stream);

// Helper: check CUDA float32 tensor, returns contiguous version
static torch::Tensor check_cuda_float32(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name + " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name + " must have dtype float32");
    return t.contiguous();
}

// MultiheadAttention forward
std::tuple<torch::Tensor, torch::Tensor> custom_multihead_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    c10::optional<torch::Tensor> in_proj_bias,
    torch::Tensor out_proj_weight,
    c10::optional<torch::Tensor> out_proj_bias,
    int64_t embed_dim,
    int64_t num_heads,
    bool need_weights,
    bool average_attn_weights) {

    // Basic checks
    TORCH_CHECK(query.dim() == 3, "query must be 3D (seq_len, batch, embed_dim)");
    TORCH_CHECK(key.dim() == 3, "key must be 3D");
    TORCH_CHECK(value.dim() == 3, "value must be 3D");
    TORCH_CHECK(in_proj_weight.dim() == 2, "in_proj_weight must be 2D");
    TORCH_CHECK(out_proj_weight.dim() == 2, "out_proj_weight must be 2D");

    // Make contiguous
    query = check_cuda_float32(query, "query");
    key = check_cuda_float32(key, "key");
    value = check_cuda_float32(value, "value");
    in_proj_weight = check_cuda_float32(in_proj_weight, "in_proj_weight");
    out_proj_weight = check_cuda_float32(out_proj_weight, "out_proj_weight");

    // Optional tensors
    torch::Tensor in_proj_bias_contig, out_proj_bias_contig;
    if (in_proj_bias.has_value() && in_proj_bias->defined()) {
        in_proj_bias_contig = check_cuda_float32(in_proj_bias.value(), "in_proj_bias");
    }
    if (out_proj_bias.has_value() && out_proj_bias->defined()) {
        out_proj_bias_contig = check_cuda_float32(out_proj_bias.value(), "out_proj_bias");
    }

    int seq_len = query.size(0);
    int batch = query.size(1);
    TORCH_CHECK(seq_len > 0 && batch > 0, "seq_len and batch must be positive");
    TORCH_CHECK(query.size(2) == embed_dim, "query last dim must equal embed_dim");
    TORCH_CHECK(key.size(0) == seq_len && key.size(1) == batch && key.size(2) == embed_dim, "key shape mismatch");
    TORCH_CHECK(value.size(0) == seq_len && value.size(1) == batch && value.size(2) == embed_dim, "value shape mismatch");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim, "in_proj_weight shape must be [3*embed_dim, embed_dim]");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim, "out_proj_weight shape must be [embed_dim, embed_dim]");

    if (in_proj_bias.has_value() && in_proj_bias->defined()) {
        TORCH_CHECK(in_proj_bias_contig.size(0) == 3 * embed_dim, "in_proj_bias size mismatch");
    }
    if (out_proj_bias.has_value() && out_proj_bias->defined()) {
        TORCH_CHECK(out_proj_bias_contig.size(0) == embed_dim, "out_proj_bias size mismatch");
    }

    // Allocate outputs
    auto attn_output = torch::empty({seq_len, batch, embed_dim}, query.options());
    torch::Tensor avg_weights;
    if (need_weights) {
        avg_weights = torch::empty({batch, seq_len, seq_len}, query.options());
    } else {
        // Return empty tensor (shape [0]) as placeholder; contract expects tuple always
        avg_weights = torch::empty({0}, query.options());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_multihead_attention_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias_contig.defined() ? in_proj_bias_contig.data_ptr<float>() : nullptr,
        out_proj_weight.data_ptr<float>(),
        out_proj_bias_contig.defined() ? out_proj_bias_contig.data_ptr<float>() : nullptr,
        attn_output.data_ptr<float>(),
        need_weights ? avg_weights.data_ptr<float>() : nullptr,
        seq_len, batch, (int)embed_dim, (int)num_heads,
        need_weights, average_attn_weights,
        stream);

    // Error check after launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_multihead_attention kernel launch failed");

    return std::make_tuple(attn_output, avg_weights);
}

// Concatenate along last dim
at::Tensor custom_cat(torch::Tensor t1, torch::Tensor t2, int64_t dim) {
    TORCH_CHECK(dim == -1 || dim == t1.dim() - 1, "Only last-dim concatenation supported");
    TORCH_CHECK(t1.dim() == t2.dim(), "t1 and t2 must have same number of dims");
    for (int i = 0; i < t1.dim() - 1; ++i) {
        TORCH_CHECK(t1.size(i) == t2.size(i), "t1 and t2 must have same shape except concat dim");
    }

    t1 = check_cuda_float32(t1, "t1");
    t2 = check_cuda_float32(t2, "t2");

    int64_t outer_size = t1.numel() / t1.size(-1);
    int64_t inner_size = t1.size(-1);

    auto output = torch::empty_like(t1, t1.options().dtype(torch::kFloat32));
    // Actually need shape with last dim doubled
    auto output_sizes = t1.sizes().vec();
    output_sizes[output_sizes.size() - 1] *= 2;
    output = torch::empty(output_sizes, t1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_cat_launcher(
        t1.data_ptr<float>(),
        t2.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)outer_size,
        (int)inner_size,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_cat kernel launch failed");
    return output;
}

// GLU along last dim
at::Tensor custom_glu(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(dim == -1 || dim == x.dim() - 1, "Only last-dim GLU supported");
    x = check_cuda_float32(x, "x");

    int64_t outer_size = x.numel() / x.size(-1);
    int64_t inner_size = x.size(-1) / 2; // output inner size
    TORCH_CHECK(x.size(-1) % 2 == 0, "Last dim must be even for GLU");

    auto output_sizes = x.sizes().vec();
    output_sizes[output_sizes.size() - 1] = inner_size;
    auto output = torch::empty(output_sizes, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_glu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)outer_size,
        (int)inner_size,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_glu kernel launch failed");
    return output;
}

// Linear forward
at::Tensor custom_linear(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    x = check_cuda_float32(x, "x");
    weight = check_cuda_float32(weight, "weight");

    torch::Tensor bias_contig;
    if (bias.has_value() && bias->defined()) {
        bias_contig = check_cuda_float32(bias.value(), "bias");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
    }

    int64_t in_features = x.size(-1);
    TORCH_CHECK(weight.size(1) == in_features, "weight in_features mismatch");
    int64_t out_features = weight.size(0);
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias_contig.size(0) == out_features, "bias size mismatch");
    }

    int64_t outer_size = x.numel() / in_features;
    auto output_sizes = x.sizes().vec();
    output_sizes[output_sizes.size() - 1] = out_features;
    auto output = torch::empty(output_sizes, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_linear_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        (int)outer_size,
        (int)in_features,
        (int)out_features,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_linear kernel launch failed");
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_multihead_attention", &custom_multihead_attention,
          "MultiheadAttention forward",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("in_proj_weight"),
          py::arg("in_proj_bias"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"),
          py::arg("embed_dim"),
          py::arg("num_heads"),
          py::arg("need_weights") = true,
          py::arg("average_attn_weights") = true);

    m.def("custom_cat", &custom_cat,
          "Concatenate two tensors along last dimension",
          py::arg("t1"), py::arg("t2"), py::arg("dim"));

    m.def("custom_glu", &custom_glu,
          "GLU along last dimension",
          py::arg("x"), py::arg("dim"));

    m.def("custom_linear", &custom_linear,
          "Linear transformation",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
