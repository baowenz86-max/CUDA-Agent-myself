#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void fused_attention_launcher(
    const float* x, const float* y,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* q_raw, float* k_raw, float* v_raw,
    float* context, float* output,
    int B, int T, int D, int H, int repeat_interleave_times,
    float scaling_factor,
    cudaStream_t stream
);

torch::Tensor fused_attention_forward(
    torch::Tensor x, torch::Tensor y,
    torch::Tensor in_proj_weight, torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias,
    int64_t repeat_interleave_times, int64_t num_heads, double scaling_factor)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(in_proj_weight.is_cuda() && in_proj_weight.is_contiguous(), "in_proj_weight must be CUDA contiguous");
    TORCH_CHECK(in_proj_bias.is_cuda() && in_proj_bias.is_contiguous(), "in_proj_bias must be CUDA contiguous");
    TORCH_CHECK(out_proj_weight.is_cuda() && out_proj_weight.is_contiguous(), "out_proj_weight must be CUDA contiguous");
    TORCH_CHECK(out_proj_bias.is_cuda() && out_proj_bias.is_contiguous(), "out_proj_bias must be CUDA contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");

    int B = x.size(0);
    int T = x.size(1);
    int D = x.size(2);
    int H = num_heads;
    TORCH_CHECK(D % H == 0 && D / H <= 64, "d_k must be <=64");
    int L = T * repeat_interleave_times;

    auto q_raw = torch::empty({B, T, D}, x.options());
    auto k_raw = torch::empty({B, T, D}, x.options());
    auto v_raw = torch::empty({B, T, D}, x.options());
    auto context = torch::empty({B, L, D}, x.options());
    auto output = torch::empty({B, L, D}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_attention_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(), in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(), out_proj_bias.data_ptr<float>(),
        q_raw.data_ptr<float>(), k_raw.data_ptr<float>(), v_raw.data_ptr<float>(),
        context.data_ptr<float>(), output.data_ptr<float>(),
        B, T, D, H, repeat_interleave_times,
        (float)scaling_factor,
        stream
    );

    return output;
}

void register_fused_attention(pybind11::module& m) {
    m.def("fused_attention_forward", &fused_attention_forward,
          "Fused attention forward",
          py::arg("x"), py::arg("y"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("repeat_interleave_times"),
          py::arg("num_heads"),
          py::arg("scaling_factor"));
}

REGISTER_BINDING(fused_attention, register_fused_attention);
