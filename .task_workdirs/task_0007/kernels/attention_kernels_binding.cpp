#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from attention_kernels.cu
extern "C" void qkv_proj_pos_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* pos_enc,
    const float* wq,
    const float* wk,
    const float* wv,
    const float* bq,
    const float* bk,
    const float* bv,
    int B,
    int T,
    int E,
    int P,
    float* q_out,
    float* k_out,
    float* v_out,
    cudaStream_t stream
);

extern "C" void sdpa_forward_launcher(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    int B,
    int T,
    int E,
    float* out,
    cudaStream_t stream
);

extern "C" void attn_bias_baddbmm_launcher(
    const float* attn_output,
    int B,
    int T,
    int E,
    float* logits,
    cudaStream_t stream
);

extern "C" void bce_with_logits_launcher(
    const float* logits,
    const float* target,
    int B,
    int T,
    float* loss,
    cudaStream_t stream
);

// Helper to check common tensor requirements
static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

py::tuple qkv_proj_pos_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor pos_enc,
    torch::Tensor wq,
    torch::Tensor wk,
    torch::Tensor wv,
    c10::optional<torch::Tensor> bq,
    c10::optional<torch::Tensor> bk,
    c10::optional<torch::Tensor> bv
) {
    // Validate inputs
    check_cuda_float32_contiguous(query, "query");
    check_cuda_float32_contiguous(key, "key");
    check_cuda_float32_contiguous(value, "value");
    check_cuda_float32_contiguous(pos_enc, "pos_enc");
    check_cuda_float32_contiguous(wq, "wq");
    check_cuda_float32_contiguous(wk, "wk");
    check_cuda_float32_contiguous(wv, "wv");

    TORCH_CHECK(query.dim() == 3, "query must be 3D [B, T, E]");
    TORCH_CHECK(key.dim() == 3, "key must be 3D [B, T, E]");
    TORCH_CHECK(value.dim() == 3, "value must be 3D [B, T, E]");
    TORCH_CHECK(pos_enc.dim() == 3, "pos_enc must be 3D [1, P, E]");
    TORCH_CHECK(wq.dim() == 2, "wq must be 2D [E, E]");
    TORCH_CHECK(wk.dim() == 2, "wk must be 2D [E, E]");
    TORCH_CHECK(wv.dim() == 2, "wv must be 2D [E, E]");

    int B = query.size(0);
    int T = query.size(1);
    int E = query.size(2);
    int P = pos_enc.size(1);

    TORCH_CHECK(key.size(0) == B && key.size(1) == T && key.size(2) == E, "key shape mismatch");
    TORCH_CHECK(value.size(0) == B && value.size(1) == T && value.size(2) == E, "value shape mismatch");
    TORCH_CHECK(pos_enc.size(0) == 1 && pos_enc.size(1) == P && pos_enc.size(2) == E, "pos_enc shape must be [1, P, E]");
    TORCH_CHECK(P >= T, "pos_enc length P must be >= T");
    TORCH_CHECK(wq.size(0) == E && wq.size(1) == E, "wq shape must be [E, E]");
    TORCH_CHECK(wk.size(0) == E && wk.size(1) == E, "wk shape must be [E, E]");
    TORCH_CHECK(wv.size(0) == E && wv.size(1) == E, "wv shape must be [E, E]");

    // Optional bias validation
    const float* bq_ptr = nullptr;
    const float* bk_ptr = nullptr;
    const float* bv_ptr = nullptr;
    if (bq.has_value()) {
        auto bias = bq.value();
        check_cuda_float32_contiguous(bias, "bq");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == E, "bq must be 1D [E]");
        bq_ptr = bias.data_ptr<float>();
    }
    if (bk.has_value()) {
        auto bias = bk.value();
        check_cuda_float32_contiguous(bias, "bk");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == E, "bk must be 1D [E]");
        bk_ptr = bias.data_ptr<float>();
    }
    if (bv.has_value()) {
        auto bias = bv.value();
        check_cuda_float32_contiguous(bias, "bv");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == E, "bv must be 1D [E]");
        bv_ptr = bias.data_ptr<float>();
    }

    // Allocate outputs
    auto q_out = torch::empty({B, T, E}, query.options());
    auto k_out = torch::empty({B, T, E}, query.options());
    auto v_out = torch::empty({B, T, E}, query.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    qkv_proj_pos_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        pos_enc.data_ptr<float>(),
        wq.data_ptr<float>(),
        wk.data_ptr<float>(),
        wv.data_ptr<float>(),
        bq_ptr,
        bk_ptr,
        bv_ptr,
        B, T, E, P,
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        v_out.data_ptr<float>(),
        stream
    );

    return py::make_tuple(q_out, k_out, v_out);
}

torch::Tensor sdpa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask
) {
    check_cuda_float32_contiguous(q, "q");
    check_cuda_float32_contiguous(k, "k");
    check_cuda_float32_contiguous(v, "v");
    check_cuda_float32_contiguous(mask, "mask");

    TORCH_CHECK(q.dim() == 3, "q must be 3D [B, T, E]");
    TORCH_CHECK(k.dim() == 3, "k must be 3D [B, T, E]");
    TORCH_CHECK(v.dim() == 3, "v must be 3D [B, T, E]");
    TORCH_CHECK(mask.dim() == 2, "mask must be 2D [T, T]");

    int B = q.size(0);
    int T = q.size(1);
    int E = q.size(2);
    TORCH_CHECK(k.size(0) == B && k.size(1) == T && k.size(2) == E, "k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == E, "v shape mismatch");
    TORCH_CHECK(mask.size(0) == T && mask.size(1) == T, "mask shape must be [T, T]");

    auto out = torch::empty({B, T, E}, q.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sdpa_forward_launcher(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        mask.data_ptr<float>(),
        B, T, E,
        out.data_ptr<float>(),
        stream
    );

    return out;
}

torch::Tensor attn_bias_baddbmm_forward(torch::Tensor attn_output) {
    check_cuda_float32_contiguous(attn_output, "attn_output");
    TORCH_CHECK(attn_output.dim() == 3, "attn_output must be 3D [B, T, E]");

    int B = attn_output.size(0);
    int T = attn_output.size(1);
    int E = attn_output.size(2);

    auto logits = torch::empty({B, T, T}, attn_output.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    attn_bias_baddbmm_launcher(
        attn_output.data_ptr<float>(),
        B, T, E,
        logits.data_ptr<float>(),
        stream
    );

    return logits;
}

torch::Tensor bce_with_logits_forward(
    torch::Tensor logits,
    torch::Tensor target
) {
    check_cuda_float32_contiguous(logits, "logits");
    check_cuda_float32_contiguous(target, "target");

    TORCH_CHECK(logits.dim() == 3, "logits must be 3D [B, T, T]");
    TORCH_CHECK(target.dim() == 3, "target must be 3D [B, T, T]");
    TORCH_CHECK(logits.size(0) == target.size(0) &&
                logits.size(1) == target.size(1) &&
                logits.size(2) == target.size(2),
                "logits and target shape mismatch");

    int B = logits.size(0);
    int T = logits.size(1);

    // Allocate scalar loss output; we'll return a 0-dim tensor
    auto loss = torch::empty({}, logits.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bce_with_logits_launcher(
        logits.data_ptr<float>(),
        target.data_ptr<float>(),
        B, T,
        loss.data_ptr<float>(),
        stream
    );

    return loss;
}

void register_attention_kernels(pybind11::module& m) {
    m.def("qkv_proj_pos", &qkv_proj_pos_forward,
          "Project QKV with linear layers and add positional encoding",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("pos_enc"),
          py::arg("wq"),
          py::arg("wk"),
          py::arg("wv"),
          py::arg("bq") = py::none(),
          py::arg("bk") = py::none(),
          py::arg("bv") = py::none());
    m.def("sdpa", &sdpa_forward,
          "Scaled Dot-Product Attention",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"));
    m.def("attn_bias_baddbmm", &attn_bias_baddbmm_forward,
          "Compute logits = attn_output @ attn_output^T (baddbmm with zero beta)",
          py::arg("attn_output"));
    m.def("bce_with_logits", &bce_with_logits_forward,
          "Binary Cross-Entropy with logits loss",
          py::arg("logits"), py::arg("target"));
}

REGISTER_BINDING(attention_kernels, register_attention_kernels);
