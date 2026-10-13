#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mha_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    int B, int L, int E, int num_heads, bool batch_first,
    float* attn_output, float* attn_weights,
    cudaStream_t stream);

extern "C" void concat_dim2_launcher(
    const float* tensor1, const float* tensor2,
    float* out, int B, int L, int D1, int D2,
    cudaStream_t stream);

extern "C" void smooth_l1_loss_launcher(
    const float* input, const float* target,
    float* output, int numel, cudaStream_t stream);

// Helper to check CUDA tensor float32 contiguous or make contiguous
static torch::Tensor ensure_contiguous_float32_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

// MHA forward wrapper
std::vector<torch::Tensor> mha_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor in_proj_weight, torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias,
    int64_t num_heads, bool batch_first) {

    // Validate inputs
    query = ensure_contiguous_float32_cuda(query, "query");
    key = ensure_contiguous_float32_cuda(key, "key");
    value = ensure_contiguous_float32_cuda(value, "value");
    in_proj_weight = ensure_contiguous_float32_cuda(in_proj_weight, "in_proj_weight");
    in_proj_bias = ensure_contiguous_float32_cuda(in_proj_bias, "in_proj_bias");
    out_proj_weight = ensure_contiguous_float32_cuda(out_proj_weight, "out_proj_weight");
    out_proj_bias = ensure_contiguous_float32_cuda(out_proj_bias, "out_proj_bias");

    // Verify batch_first is true (only supported mode)
    TORCH_CHECK(batch_first, "Only batch_first=True is supported");

    // Get dimensions
    int64_t B, L, E;
    B = query.size(0);
    L = query.size(1);
    E = query.size(2);

    TORCH_CHECK(key.size(0) == B && key.size(1) == L && key.size(2) == E, "key shape mismatch");
    TORCH_CHECK(value.size(0) == B && value.size(1) == L && value.size(2) == E, "value shape mismatch");
    TORCH_CHECK(in_proj_weight.size(0) == 3 * E && in_proj_weight.size(1) == E, "in_proj_weight shape mismatch");
    TORCH_CHECK(in_proj_bias.size(0) == 3 * E, "in_proj_bias size mismatch");
    TORCH_CHECK(out_proj_weight.size(0) == E && out_proj_weight.size(1) == E, "out_proj_weight shape mismatch");
    TORCH_CHECK(out_proj_bias.size(0) == E, "out_proj_bias size mismatch");
    TORCH_CHECK(num_heads > 0 && E % num_heads == 0, "num_heads must divide embed_dim");

    // Allocate output tensors
    auto attn_output = torch::empty({B, L, E}, query.options());
    auto attn_weights = torch::empty({B, L, L}, query.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mha_forward_launcher(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(), in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(), out_proj_bias.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(L), static_cast<int>(E),
        static_cast<int>(num_heads), batch_first,
        attn_output.data_ptr<float>(), attn_weights.data_ptr<float>(),
        stream);

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mha_forward_launcher failed with CUDA error: ", cudaGetErrorString(err));

    return {attn_output, attn_weights};
}

// concat_dim2 wrapper
torch::Tensor concat_dim2(torch::Tensor tensor1, torch::Tensor tensor2) {
    tensor1 = ensure_contiguous_float32_cuda(tensor1, "tensor1");
    tensor2 = ensure_contiguous_float32_cuda(tensor2, "tensor2");

    int64_t B = tensor1.size(0);
    int64_t L = tensor1.size(1);
    int64_t D1 = tensor1.size(2);
    int64_t D2 = tensor2.size(2);

    TORCH_CHECK(tensor2.size(0) == B && tensor2.size(1) == L, "tensor2 batch or length mismatch");

    auto out = torch::empty({B, L, D1 + D2}, tensor1.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    concat_dim2_launcher(
        tensor1.data_ptr<float>(), tensor2.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(L),
        static_cast<int>(D1), static_cast<int>(D2),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "concat_dim2_launcher failed with CUDA error: ", cudaGetErrorString(err));

    return out;
}

// smooth_l1_loss wrapper
torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target) {
    input = ensure_contiguous_float32_cuda(input, "input");
    target = ensure_contiguous_float32_cuda(target, "target");

    TORCH_CHECK(input.sizes() == target.sizes(), "input and target shapes must match");

    int64_t numel = input.numel();
    auto loss = torch::empty({}, input.options());  // scalar tensor

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    smooth_l1_loss_launcher(
        input.data_ptr<float>(), target.data_ptr<float>(),
        loss.data_ptr<float>(),
        static_cast<int>(numel),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "smooth_l1_loss_launcher failed with CUDA error: ", cudaGetErrorString(err));

    return loss;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mha_forward", &mha_forward,
          "MultiheadAttention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"),
          py::arg("out_proj_weight"), py::arg("out_proj_bias"),
          py::arg("num_heads"), py::arg("batch_first"));

    m.def("concat_dim2", &concat_dim2,
          "Concatenate two tensors along dimension 2",
          py::arg("tensor1"), py::arg("tensor2"));

    m.def("smooth_l1_loss", &smooth_l1_loss,
          "Smooth L1 loss (mean reduction)",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
