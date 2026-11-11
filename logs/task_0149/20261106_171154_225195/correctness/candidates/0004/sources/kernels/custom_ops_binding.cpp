#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void view_as_complex_launcher(
    const float* input,
    float2* output,
    long long total_complex,
    cudaStream_t stream);

extern "C" void multihead_attention_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* output,
    float* attn_weights,
    float* q_buf,
    float* k_buf,
    float* v_buf,
    float* scores_buf,
    float* context_buf,
    float* qk_proj_buf,
    float* v_proj_buf,
    int B,
    int S,
    int E,
    int H,
    int head_dim,
    cudaStream_t stream);

// view_as_complex binding
static torch::Tensor view_as_complex_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "view_as_complex: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "view_as_complex: input must be float32");
    TORCH_CHECK(input.dim() == 4, "view_as_complex: input must be 4D");
    TORCH_CHECK(input.size(-1) == 2, "view_as_complex: last dimension must be 2");

    // Ensure contiguous (kernel assumes contiguous)
    auto input_contig = input.contiguous();
    auto sizes = input_contig.sizes().vec();
    sizes.pop_back(); // remove last dim of size 2
    auto output = torch::empty(sizes, torch::TensorOptions().dtype(torch::kComplexFloat).device(input.device()));

    long long total_complex = output.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    view_as_complex_launcher(
        input_contig.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>()),
        total_complex,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "view_as_complex kernel failed: ", cudaGetErrorString(err));
    return output;
}

// multihead_attention binding
static torch::Tensor multihead_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t num_heads,
    bool batch_first) {

    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
                "multihead_attention: query, key, value must be CUDA tensors");
    TORCH_CHECK(in_proj_weight.is_cuda() && in_proj_bias.is_cuda() &&
                out_proj_weight.is_cuda() && out_proj_bias.is_cuda(),
                "multihead_attention: projection tensors must be CUDA");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32 &&
                value.dtype() == torch::kFloat32, "multihead_attention: inputs must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32 && in_proj_bias.dtype() == torch::kFloat32 &&
                out_proj_weight.dtype() == torch::kFloat32 && out_proj_bias.dtype() == torch::kFloat32,
                "multihead_attention: projections must be float32");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
                "multihead_attention: query, key, value must be 3D");
    TORCH_CHECK(batch_first, "multihead_attention: only batch_first=True is supported");

    int B, S, E;
    B = query.size(0);
    S = query.size(1);
    E = query.size(2);
    TORCH_CHECK(key.size(0) == B && key.size(1) == S && key.size(2) == E,
                "multihead_attention: query, key dimensions mismatch");
    TORCH_CHECK(value.size(0) == B && value.size(1) == S && value.size(2) == E,
                "multihead_attention: value dimensions mismatch");
    TORCH_CHECK(E % num_heads == 0, "multihead_attention: embed_dim must be divisible by num_heads");
    int head_dim = E / num_heads;

    // Ensure contiguous
    auto query_contig = query.contiguous();
    auto key_contig = key.contiguous();
    auto value_contig = value.contiguous();
    auto in_proj_weight_contig = in_proj_weight.contiguous();
    auto in_proj_bias_contig = in_proj_bias.contiguous();
    auto out_proj_weight_contig = out_proj_weight.contiguous();
    auto out_proj_bias_contig = out_proj_bias.contiguous();

    // Allocate output and intermediate buffers
    auto output = torch::empty({B, S, E}, query.options());
    auto attn_weights = torch::empty({B, S, S}, query.options());
    auto q_buf = torch::empty({B * num_heads * S * head_dim}, query.options());
    auto k_buf = torch::empty({B * num_heads * S * head_dim}, query.options());
    auto v_buf = torch::empty({B * num_heads * S * head_dim}, query.options());
    auto scores_buf = torch::empty({B * num_heads * S * S}, query.options());
    auto context_buf = torch::empty({B * num_heads * S * head_dim}, query.options());
    auto qk_proj_buf = torch::empty({B * S * 2 * E}, query.options());
    auto v_proj_buf = torch::empty({B * S * E}, query.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    multihead_attention_launcher(
        query_contig.data_ptr<float>(),
        key_contig.data_ptr<float>(),
        value_contig.data_ptr<float>(),
        in_proj_weight_contig.data_ptr<float>(),
        in_proj_bias_contig.data_ptr<float>(),
        out_proj_weight_contig.data_ptr<float>(),
        out_proj_bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        q_buf.data_ptr<float>(),
        k_buf.data_ptr<float>(),
        v_buf.data_ptr<float>(),
        scores_buf.data_ptr<float>(),
        context_buf.data_ptr<float>(),
        qk_proj_buf.data_ptr<float>(),
        v_proj_buf.data_ptr<float>(),
        B, S, E, static_cast<int>(num_heads), head_dim,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "multihead_attention kernel failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("view_as_complex", &view_as_complex_forward, "view_as_complex forward",
          py::arg("input"));
    m.def("multihead_attention", &multihead_attention_forward, "multihead_attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("num_heads"), py::arg("batch_first"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
