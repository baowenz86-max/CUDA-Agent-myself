// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* attn_output_weights,
    int B, int S, int E, int num_heads, int batch_first,
    cudaStream_t stream);

extern "C" void concat_dim2_launcher(
    const float* a, const float* b, float* out,
    int B, int S, int E1, int E2,
    cudaStream_t stream);

extern "C" void smooth_l1_loss_launcher(
    const float* input, const float* target, float beta,
    int N, float* output, cudaStream_t stream);

// Helper to check CUDA tensor requirements
void check_cuda_float_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// multihead_attention_forward binding
torch::Tensor multihead_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    int64_t num_heads,
    bool batch_first)
{
    // Validate inputs
    check_cuda_float_contiguous(query, "query");
    check_cuda_float_contiguous(key, "key");
    check_cuda_float_contiguous(value, "value");
    check_cuda_float_contiguous(in_proj_weight, "in_proj_weight");
    check_cuda_float_contiguous(in_proj_bias, "in_proj_bias");
    check_cuda_float_contiguous(out_proj_weight, "out_proj_weight");
    check_cuda_float_contiguous(out_proj_bias, "out_proj_bias");

    TORCH_CHECK(batch_first, "Only batch_first=True is supported");

    // Extract dimensions
    auto sizes = query.sizes();
    TORCH_CHECK(sizes.size() == 3, "query must be 3D [B, S, E]");
    int64_t B = sizes[0];
    int64_t S = sizes[1];
    int64_t E = sizes[2];
    TORCH_CHECK(key.sizes() == sizes, "key must have same shape as query");
    TORCH_CHECK(value.sizes() == sizes, "value must have same shape as query");
    TORCH_CHECK(E % num_heads == 0, "embed_dim must be divisible by num_heads");

    TORCH_CHECK(in_proj_weight.sizes() == torch::IntArrayRef({3 * E, E}),
                "in_proj_weight must have shape [3*E, E]");
    TORCH_CHECK(in_proj_bias.sizes() == torch::IntArrayRef({3 * E}),
                "in_proj_bias must have shape [3*E]");
    TORCH_CHECK(out_proj_weight.sizes() == torch::IntArrayRef({E, E}),
                "out_proj_weight must have shape [E, E]");
    TORCH_CHECK(out_proj_bias.sizes() == torch::IntArrayRef({E}),
                "out_proj_bias must have shape [E]");

    // Allocate outputs
    auto attn_output = torch::empty({B, S, E}, query.options());
    auto attn_output_weights = torch::empty({B, S, S}, query.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    multihead_attention_forward_launcher(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(), in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(), out_proj_bias.data_ptr<float>(),
        attn_output.data_ptr<float>(), attn_output_weights.data_ptr<float>(),
        (int)B, (int)S, (int)E, (int)num_heads, (int)batch_first,
        stream);

    // Return tuple (attn_output, attn_output_weights)
    return py::make_tuple(attn_output, attn_output_weights);
}

// concat_dim2 binding
torch::Tensor concat_dim2(torch::Tensor tensor1, torch::Tensor tensor2)
{
    check_cuda_float_contiguous(tensor1, "tensor1");
    check_cuda_float_contiguous(tensor2, "tensor2");

    auto sizes1 = tensor1.sizes();
    auto sizes2 = tensor2.sizes();
    TORCH_CHECK(sizes1.size() == 3 && sizes2.size() == 3, "Both tensors must be 3D");
    TORCH_CHECK(sizes1[0] == sizes2[0] && sizes1[1] == sizes2[1],
                "Batch and seq dimensions must match");

    int64_t B = sizes1[0];
    int64_t S = sizes1[1];
    int64_t E1 = sizes1[2];
    int64_t E2 = sizes2[2];

    auto output = torch::empty({B, S, E1 + E2}, tensor1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    concat_dim2_launcher(
        tensor1.data_ptr<float>(), tensor2.data_ptr<float>(), output.data_ptr<float>(),
        (int)B, (int)S, (int)E1, (int)E2, stream);

    return output;
}

// smooth_l1_loss binding
torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, double beta)
{
    check_cuda_float_contiguous(input, "input");
    check_cuda_float_contiguous(target, "target");

    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");
    TORCH_CHECK(beta > 0, "beta must be positive");

    int64_t N = input.numel();
    // Allocate scalar output on device
    auto output = torch::empty({}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    smooth_l1_loss_launcher(
        input.data_ptr<float>(), target.data_ptr<float>(), (float)beta,
        (int)N, output.data_ptr<float>(), stream);

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multihead attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("num_heads"), py::arg("batch_first"));
    m.def("concat_dim2", &concat_dim2, "Concatenate along dim=2",
          py::arg("tensor1"), py::arg("tensor2"));
    m.def("smooth_l1_loss", &smooth_l1_loss, "Smooth L1 loss",
          py::arg("input"), py::arg("target"), py::arg("beta") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
