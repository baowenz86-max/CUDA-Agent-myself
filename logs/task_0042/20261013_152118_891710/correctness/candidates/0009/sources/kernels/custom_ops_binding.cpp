#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu (to be implemented)
extern "C" void multihead_attn_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* attn_output,
    float* attn_output_weights,
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    bool batch_first,
    cudaStream_t stream
);

extern "C" void cat_launcher(
    const float* a,
    const float* b,
    float* out,
    int batch_size,
    int seq_len,
    int embed_dim,
    int dim,
    cudaStream_t stream
);

extern "C" void smooth_l1_loss_launcher(
    const float* input,
    const float* target,
    float* out,
    int64_t numel,
    cudaStream_t stream
);

static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

std::tuple<torch::Tensor, torch::Tensor> multihead_attn_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t num_heads,
    bool batch_first
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be a CUDA tensor");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be a CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be a CUDA tensor");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be a CUDA tensor");

    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(in_proj_bias.dtype() == torch::kFloat32, "in_proj_bias must be float32");
    TORCH_CHECK(out_proj_weight.dtype() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(out_proj_bias.dtype() == torch::kFloat32, "out_proj_bias must be float32");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(in_proj_weight.is_contiguous(), "in_proj_weight must be contiguous");
    TORCH_CHECK(in_proj_bias.is_contiguous(), "in_proj_bias must be contiguous");
    TORCH_CHECK(out_proj_weight.is_contiguous(), "out_proj_weight must be contiguous");
    TORCH_CHECK(out_proj_bias.is_contiguous(), "out_proj_bias must be contiguous");

    int batch_size, seq_len, embed_dim;
    if (batch_first) {
        TORCH_CHECK(query.dim() == 3, "query must be 3D when batch_first=True");
        batch_size = query.size(0);
        seq_len = query.size(1);
        embed_dim = query.size(2);
    } else {
        TORCH_CHECK(query.dim() == 3, "query must be 3D when batch_first=False");
        seq_len = query.size(0);
        batch_size = query.size(1);
        embed_dim = query.size(2);
    }
    TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

    TORCH_CHECK(key.sizes() == query.sizes(), "key must have same shape as query");
    TORCH_CHECK(value.sizes() == query.sizes(), "value must have same shape as query");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * embed_dim && in_proj_weight.size(1) == embed_dim,
                "in_proj_weight shape must be [3*embed_dim, embed_dim]");
    TORCH_CHECK(in_proj_bias.numel() == 3 * embed_dim, "in_proj_bias size must be 3*embed_dim");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim && out_proj_weight.size(1) == embed_dim,
                "out_proj_weight shape must be [embed_dim, embed_dim]");
    TORCH_CHECK(out_proj_bias.numel() == embed_dim, "out_proj_bias size must be embed_dim");

    auto attn_output = torch::empty({batch_size, seq_len, embed_dim}, query.options());
    auto attn_output_weights = torch::empty({batch_size, seq_len, seq_len}, query.options());

    cudaStream_t stream = getCurrentCUDAStream();

    multihead_attn_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        attn_output_weights.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim,
        num_heads,
        batch_first,
        stream
    );

    return std::make_tuple(attn_output, attn_output_weights);
}

torch::Tensor cat_forward(torch::Tensor a, torch::Tensor b, int64_t dim) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    TORCH_CHECK(a.dim() == 3, "a must be 3D");
    TORCH_CHECK(b.dim() == 3, "b must be 3D");
    TORCH_CHECK(dim == 2, "only dim=2 is supported");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(1) == b.size(1) && a.size(2) == b.size(2),
                "a and b must have same shape except dim");

    int batch_size = a.size(0);
    int seq_len = a.size(1);
    int embed_dim = a.size(2);

    auto out = torch::empty({batch_size, seq_len, 2 * embed_dim}, a.options());

    cudaStream_t stream = getCurrentCUDAStream();

    cat_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim,
        dim,
        stream
    );

    return out;
}

torch::Tensor smooth_l1_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");

    int64_t numel = input.numel();
    auto out = torch::empty({}, input.options());

    cudaStream_t stream = getCurrentCUDAStream();

    smooth_l1_loss_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("multihead_attn_forward", &multihead_attn_forward,
          "Multihead attention forward",
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("in_proj_weight"),
          pybind11::arg("in_proj_bias"),
          pybind11::arg("out_proj_weight"),
          pybind11::arg("out_proj_bias"),
          pybind11::arg("num_heads"),
          pybind11::arg("batch_first"));

    m.def("cat_forward", &cat_forward,
          "Concatenate two tensors along dim 2",
          pybind11::arg("a"),
          pybind11::arg("b"),
          pybind11::arg("dim"));

    m.def("smooth_l1_loss_forward", &smooth_l1_loss_forward,
          "Smooth L1 loss forward",
          pybind11::arg("input"),
          pybind11::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
