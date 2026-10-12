#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void mha_forward_launcher(
    const float* x1, const float* x2, const float* x3,
    const bool* key_padding_mask,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output,
    float* Q, float* K, float* V,
    float* q_t, float* k_t, float* v_t,
    float* merged_out,
    float* out_proj,
    int B, int S, int E, int H, int stack_dim,
    cudaStream_t stream
);

torch::Tensor mha_stack_forward(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor x3,
    torch::Tensor key_padding_mask,
    torch::Tensor in_proj_weight, torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias,
    int64_t num_heads, int64_t stack_dim
) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be CUDA");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_cuda() && x2.dtype() == torch::kFloat32 && x2.is_contiguous(), "x2 invalid");
    TORCH_CHECK(x3.is_cuda() && x3.dtype() == torch::kFloat32 && x3.is_contiguous(), "x3 invalid");
    TORCH_CHECK(key_padding_mask.is_cuda() && key_padding_mask.dtype() == torch::kBool && key_padding_mask.is_contiguous(), "mask invalid");
    TORCH_CHECK(in_proj_weight.is_cuda() && in_proj_weight.dtype() == torch::kFloat32 && in_proj_weight.is_contiguous(), "in_proj_weight invalid");
    TORCH_CHECK(in_proj_bias.is_cuda() && in_proj_bias.dtype() == torch::kFloat32 && in_proj_bias.is_contiguous(), "in_proj_bias invalid");
    TORCH_CHECK(out_proj_weight.is_cuda() && out_proj_weight.dtype() == torch::kFloat32 && out_proj_weight.is_contiguous(), "out_proj_weight invalid");
    TORCH_CHECK(out_proj_bias.is_cuda() && out_proj_bias.dtype() == torch::kFloat32 && out_proj_bias.is_contiguous(), "out_proj_bias invalid");

    int B = x1.size(0);
    int S = x1.size(1);
    int E = x1.size(2);
    int H = num_heads;
    int D = E / H;
    TORCH_CHECK(E % H == 0, "embed_dim must be divisible by num_heads");
    TORCH_CHECK(x2.size(0) == B && x2.size(1) == S && x2.size(2) == E, "x2 shape mismatch");
    TORCH_CHECK(x3.size(0) == B && x3.size(1) == S && x3.size(2) == E, "x3 shape mismatch");
    TORCH_CHECK(key_padding_mask.size(0) == B && key_padding_mask.size(1) == S, "mask shape mismatch");
    TORCH_CHECK(in_proj_weight.size(0) == 3*E && in_proj_weight.size(1) == E, "in_proj_weight shape");
    TORCH_CHECK(in_proj_bias.size(0) == 3*E, "in_proj_bias size");
    TORCH_CHECK(out_proj_weight.size(0) == E && out_proj_weight.size(1) == E, "out_proj_weight shape");
    TORCH_CHECK(out_proj_bias.size(0) == E, "out_proj_bias size");

    int stack_dim_int = static_cast<int>(stack_dim);
    int out_d0, out_d1, out_d2, out_d3;
    if (stack_dim_int < 0) stack_dim_int += 4;
    if (stack_dim_int == 0) { out_d0=2; out_d1=B; out_d2=S; out_d3=E; }
    else if (stack_dim_int == 1) { out_d0=B; out_d1=2; out_d2=S; out_d3=E; }
    else if (stack_dim_int == 2) { out_d0=B; out_d1=S; out_d2=2; out_d3=E; }
    else if (stack_dim_int == 3) { out_d0=B; out_d1=S; out_d2=E; out_d3=2; }
    else TORCH_CHECK(false, "invalid stack_dim");

    auto options = x1.options();
    auto Q = torch::empty({B*S, E}, options);
    auto K = torch::empty({B*S, E}, options);
    auto V = torch::empty({B*S, E}, options);
    auto q_t = torch::empty({B, H, S, D}, options);
    auto k_t = torch::empty({B, H, S, D}, options);
    auto v_t = torch::empty({B, H, S, D}, options);
    auto merged_out = torch::empty({B*S, E}, options);
    auto out_proj = torch::empty({B*S, E}, options);
    auto output = torch::empty({out_d0, out_d1, out_d2, out_d3}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mha_forward_launcher(
        x1.data_ptr<float>(), x2.data_ptr<float>(), x3.data_ptr<float>(),
        key_padding_mask.data_ptr<bool>(),
        in_proj_weight.data_ptr<float>(), in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(), out_proj_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        q_t.data_ptr<float>(), k_t.data_ptr<float>(), v_t.data_ptr<float>(),
        merged_out.data_ptr<float>(),
        out_proj.data_ptr<float>(),
        B, S, E, H, stack_dim_int,
        stream
    );

    return output;
}

void register_mha_stack(pybind11::module& m) {
    m.def("mha_stack_forward", &mha_stack_forward,
          "MultiheadAttention + multiply + stack",
          py::arg("x1"), py::arg("x2"), py::arg("x3"), py::arg("key_padding_mask"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("num_heads"), py::arg("stack_dim"));
}

REGISTER_BINDING(mha_stack, register_mha_stack);
