// Binding for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* attn_weights,
    int B, int T, int E, int num_heads,
    bool batch_first, bool add_zero_attn, bool is_causal,
    const float* key_padding_mask, const float* attn_mask,
    bool need_weights, bool average_attn_weights,
    cudaStream_t stream
);

extern "C" void matmul_forward_launcher(
    const float* a, const float* b, float* out,
    int B, int M, int K, int N,
    int a_b_stride, int a_m_stride, int a_k_stride,
    int b_b_stride, int b_k_stride, int b_n_stride,
    int out_b_stride, int out_m_stride, int out_n_stride,
    cudaStream_t stream
);

// Helper to check CUDA tensor and convert to contiguous float if needed
torch::Tensor check_contiguous_float_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

py::tuple multihead_attention_forward(
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
    bool add_zero_attn,
    bool is_causal,
    c10::optional<torch::Tensor> key_padding_mask,
    c10::optional<torch::Tensor> attn_mask,
    bool need_weights,
    bool average_attn_weights
) {
    // Basic checks
    TORCH_CHECK(batch_first, "Only batch_first=True is supported");
    TORCH_CHECK(!add_zero_attn, "add_zero_attn is not supported");
    TORCH_CHECK(!is_causal, "is_causal is not supported");
    TORCH_CHECK(!key_padding_mask.has_value(), "key_padding_mask is not supported");
    TORCH_CHECK(!attn_mask.has_value(), "attn_mask is not supported");

    // Check and make contiguous
    query = check_contiguous_float_cuda(query, "query");
    key = check_contiguous_float_cuda(key, "key");
    value = check_contiguous_float_cuda(value, "value");
    in_proj_weight = check_contiguous_float_cuda(in_proj_weight, "in_proj_weight");
    in_proj_bias = check_contiguous_float_cuda(in_proj_bias, "in_proj_bias");
    out_proj_weight = check_contiguous_float_cuda(out_proj_weight, "out_proj_weight");
    out_proj_bias = check_contiguous_float_cuda(out_proj_bias, "out_proj_bias");

    // Verify dimensions
    TORCH_CHECK(query.dim() == 3, "query must be 3D [B,T,E]");
    TORCH_CHECK(key.dim() == 3, "key must be 3D");
    TORCH_CHECK(value.dim() == 3, "value must be 3D");
    int64_t B = query.size(0);
    int64_t T = query.size(1);
    int64_t E = query.size(2);
    TORCH_CHECK(E == embed_dim, "embed_dim mismatch");
    TORCH_CHECK(key.size(0) == B && key.size(1) == T && key.size(2) == E, "key shape mismatch");
    TORCH_CHECK(value.size(0) == B && value.size(1) == T && value.size(2) == E, "value shape mismatch");

    TORCH_CHECK(in_proj_weight.dim() == 2 && in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim, "in_proj_weight shape mismatch");
    TORCH_CHECK(in_proj_bias.dim() == 1 && in_proj_bias.size(0) == 3 * embed_dim, "in_proj_bias shape mismatch");
    TORCH_CHECK(out_proj_weight.dim() == 2 && out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim, "out_proj_weight shape mismatch");
    TORCH_CHECK(out_proj_bias.dim() == 1 && out_proj_bias.size(0) == embed_dim, "out_proj_bias shape mismatch");

    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
    int64_t head_dim = embed_dim / num_heads;

    // Allocate outputs
    auto attn_output = torch::empty({B, T, embed_dim}, query.options());
    auto attn_weights = torch::empty({B, T, T}, query.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    multihead_attention_forward_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        (int)B, (int)T, (int)embed_dim, (int)num_heads,
        batch_first, add_zero_attn, is_causal,
        nullptr, // key_padding_mask not used
        nullptr, // attn_mask not used
        need_weights, average_attn_weights,
        stream
    );

    // Check for kernel errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in multihead_attention_forward");

    return py::make_tuple(attn_output, attn_weights);
}

torch::Tensor matmul_forward(
    torch::Tensor a,
    torch::Tensor b
) {
    a = check_contiguous_float_cuda(a, "a");
    b = check_contiguous_float_cuda(b, "b");
    TORCH_CHECK(a.dim() == 3, "a must be 3D [B,M,K]");
    TORCH_CHECK(b.dim() == 3, "b must be 3D [B,K,N]");
    int64_t B = a.size(0);
    int64_t M = a.size(1);
    int64_t K = a.size(2);
    int64_t N = b.size(2);
    TORCH_CHECK(b.size(0) == B, "batch size mismatch");
    TORCH_CHECK(b.size(1) == K, "inner dimension mismatch");

    auto out = torch::empty({B, M, N}, a.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides after contiguity: a has strides [M*K, K, 1], b has [K*N, N, 1], out [M*N, N, 1]
    int64_t a_b_stride = M * K;
    int64_t a_m_stride = K;
    int64_t a_k_stride = 1;
    int64_t b_b_stride = K * N;
    int64_t b_k_stride = N;
    int64_t b_n_stride = 1;
    int64_t out_b_stride = M * N;
    int64_t out_m_stride = N;
    int64_t out_n_stride = 1;

    matmul_forward_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)B, (int)M, (int)K, (int)N,
        (int)a_b_stride, (int)a_m_stride, (int)a_k_stride,
        (int)b_b_stride, (int)b_k_stride, (int)b_n_stride,
        (int)out_b_stride, (int)out_m_stride, (int)out_n_stride,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in matmul_forward");

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multi-head attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("embed_dim") = 128, py::arg("num_heads") = 4,
          py::arg("batch_first") = true, py::arg("add_zero_attn") = false,
          py::arg("is_causal") = false,
          py::arg("key_padding_mask") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("need_weights") = true,
          py::arg("average_attn_weights") = true);

    m.def("matmul_forward", &matmul_forward,
          "Batched matrix multiplication",
          py::arg("a"), py::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
