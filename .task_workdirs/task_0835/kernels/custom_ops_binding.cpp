// binding for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void multihead_attn_forward(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    int64_t batch_size, int64_t seq_len, int64_t embed_dim, int64_t num_heads,
    bool batch_first,
    float* attn_output, float* attn_output_weights,
    cudaStream_t stream);

extern "C" void log10_abs_eps_launcher(
    const float* x, float* out, long long numel, double eps, cudaStream_t stream);

// Helper to check CUDA errors after launcher calls
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Wrapper for multihead attention
torch::Tensor multihead_attn_forward_wrapper(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor in_proj_weight, torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias,
    int64_t embed_dim, int64_t num_heads, bool batch_first) {

    // Contract: all tensors are float32, CUDA
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be CUDA tensor");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be CUDA tensor");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be CUDA tensor");

    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.scalar_type() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(in_proj_weight.scalar_type() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(in_proj_bias.scalar_type() == torch::kFloat32, "in_proj_bias must be float32");
    TORCH_CHECK(out_proj_weight.scalar_type() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(out_proj_bias.scalar_type() == torch::kFloat32, "out_proj_bias must be float32");

    // Infer batch, seq_len from query shape
    int64_t ndims = query.dim();
    TORCH_CHECK(ndims == 3, "query must be 3D");
    int64_t batch_size, seq_len;
    if (batch_first) {
        batch_size = query.size(0);
        seq_len = query.size(1);
        TORCH_CHECK(query.size(2) == embed_dim, "query last dim must match embed_dim");
    } else {
        seq_len = query.size(0);
        batch_size = query.size(1);
        TORCH_CHECK(query.size(2) == embed_dim, "query last dim must match embed_dim");
    }
    // Validate key and value shapes (same as query)
    TORCH_CHECK(key.sizes() == query.sizes(), "key shape mismatch");
    TORCH_CHECK(value.sizes() == query.sizes(), "value shape mismatch");

    // Validate projection parameter shapes
    int64_t three_embed = 3 * embed_dim;
    TORCH_CHECK(in_proj_weight.size(0) == three_embed && in_proj_weight.size(1) == embed_dim,
                "in_proj_weight shape must be [3*embed_dim, embed_dim]");
    TORCH_CHECK(in_proj_bias.size(0) == three_embed, "in_proj_bias shape must be [3*embed_dim]");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim,
                "out_proj_weight shape must be [embed_dim, embed_dim]");
    TORCH_CHECK(out_proj_bias.size(0) == embed_dim, "out_proj_bias shape must be [embed_dim]");
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

    // Ensure contiguous tensors for launcher (launcher expects contiguous pointers)
    auto query_contig = query.contiguous();
    auto key_contig = key.contiguous();
    auto value_contig = value.contiguous();
    auto in_proj_weight_contig = in_proj_weight.contiguous();
    auto in_proj_bias_contig = in_proj_bias.contiguous();
    auto out_proj_weight_contig = out_proj_weight.contiguous();
    auto out_proj_bias_contig = out_proj_bias.contiguous();

    // Allocate output tensors
    auto attn_output = torch::empty_like(query_contig);
    // attn_output_weights shape: [batch, seq, seq] if batch_first else [seq, batch, batch]
    std::vector<int64_t> weights_shape;
    if (batch_first) {
        weights_shape = {batch_size, seq_len, seq_len};
    } else {
        weights_shape = {seq_len, batch_size, batch_size};
    }
    auto attn_output_weights = torch::empty(weights_shape, query_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    multihead_attn_forward(
        query_contig.data_ptr<float>(),
        key_contig.data_ptr<float>(),
        value_contig.data_ptr<float>(),
        in_proj_weight_contig.data_ptr<float>(),
        in_proj_bias_contig.data_ptr<float>(),
        out_proj_weight_contig.data_ptr<float>(),
        out_proj_bias_contig.data_ptr<float>(),
        batch_size, seq_len, embed_dim, num_heads,
        batch_first,
        attn_output.data_ptr<float>(),
        attn_output_weights.data_ptr<float>(),
        stream
    );
    check_cuda_error("multihead_attn_forward kernel failed");

    return std::make_tuple(attn_output, attn_output_weights);
}

// Wrapper for log10_abs_eps
torch::Tensor log10_abs_eps_wrapper(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log10_abs_eps_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        numel, // launcher expects long long, int64_t converts
        eps,
        stream
    );
    check_cuda_error("log10_abs_eps kernel failed");
    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("multihead_attn_forward", &multihead_attn_forward_wrapper,
          "Multihead attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("embed_dim"), py::arg("num_heads"), py::arg("batch_first"));
    m.def("log10_abs_eps", &log10_abs_eps_wrapper,
          "log10(abs(x)+eps)",
          py::arg("x"), py::arg("eps"));
}

// Register binding
REGISTER_BINDING(custom_ops, register_custom_ops);
