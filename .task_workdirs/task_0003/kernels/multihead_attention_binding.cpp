#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Launcher from multihead_attention.cu
extern "C" void multihead_attention_forward_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    const bool* padding_mask,
    float* output,
    int B, int T, int E, int H,
    bool batch_first,
    bool use_bias,
    bool mask_present,
    cudaStream_t stream
);

// Multihead attention forward wrapper
torch::Tensor multihead_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor in_proj_weight,
    c10::optional<torch::Tensor> in_proj_bias_opt,
    torch::Tensor out_proj_weight,
    c10::optional<torch::Tensor> out_proj_bias_opt,
    torch::Tensor key_padding_mask,
    int64_t embed_dim,
    int64_t num_heads,
    bool batch_first
) {
    // Check devices
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be a CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be a CUDA tensor");
    TORCH_CHECK(key_padding_mask.is_cuda(), "key_padding_mask must be a CUDA tensor");

    // Check dtypes
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(out_proj_weight.dtype() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(key_padding_mask.dtype() == torch::kBool, "key_padding_mask must be bool");

    // Check contiguous
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(in_proj_weight.is_contiguous(), "in_proj_weight must be contiguous");
    TORCH_CHECK(out_proj_weight.is_contiguous(), "out_proj_weight must be contiguous");
    TORCH_CHECK(key_padding_mask.is_contiguous(), "key_padding_mask must be contiguous");

    // Optional biases
    bool use_bias = in_proj_bias_opt.has_value() && out_proj_bias_opt.has_value();
    TORCH_CHECK(use_bias == (in_proj_bias_opt.has_value() && out_proj_bias_opt.has_value()),
                "in_proj_bias and out_proj_bias must both be provided or both be None");
    if (use_bias) {
        TORCH_CHECK(in_proj_bias_opt.value().is_cuda(), "in_proj_bias must be CUDA");
        TORCH_CHECK(out_proj_bias_opt.value().is_cuda(), "out_proj_bias must be CUDA");
        TORCH_CHECK(in_proj_bias_opt.value().dtype() == torch::kFloat32, "in_proj_bias must be float32");
        TORCH_CHECK(out_proj_bias_opt.value().dtype() == torch::kFloat32, "out_proj_bias must be float32");
        TORCH_CHECK(in_proj_bias_opt.value().is_contiguous(), "in_proj_bias must be contiguous");
        TORCH_CHECK(out_proj_bias_opt.value().is_contiguous(), "out_proj_bias must be contiguous");
    }

    // Derive dimensions.
    // query shape: (B, T, E) if batch_first else (T, B, E).
    int64_t B, T, E;
    if (batch_first) {
        TORCH_CHECK(query.dim() == 3, "batch_first: query must be 3D (B, T, E)");
        B = query.size(0);
        T = query.size(1);
        E = query.size(2);
    } else {
        TORCH_CHECK(query.dim() == 3, "non-batch_first: query must be 3D (T, B, E)");
        T = query.size(0);
        B = query.size(1);
        E = query.size(2);
    }
    TORCH_CHECK(E == embed_dim, "embed_dim mismatch: query feature dim != embed_dim");
    TORCH_CHECK(num_heads > 0 && E % num_heads == 0, "embed_dim must be divisible by num_heads");

    // Check key and value shapes match query
    if (batch_first) {
        TORCH_CHECK(key.dim() == 3 && key.size(0) == B && key.size(1) == T && key.size(2) == E,
                    "key shape mismatch with query");
        TORCH_CHECK(value.dim() == 3 && value.size(0) == B && value.size(1) == T && value.size(2) == E,
                    "value shape mismatch with query");
    } else {
        TORCH_CHECK(key.dim() == 3 && key.size(0) == T && key.size(1) == B && key.size(2) == E,
                    "key shape mismatch with query");
        TORCH_CHECK(value.dim() == 3 && value.size(0) == T && value.size(1) == B && value.size(2) == E,
                    "value shape mismatch with query");
    }

    // Check mask shape: (B, T)
    TORCH_CHECK(key_padding_mask.dim() == 2, "key_padding_mask must be 2D");
    TORCH_CHECK(key_padding_mask.size(0) == B, "key_padding_mask batch dim mismatch");
    TORCH_CHECK(key_padding_mask.size(1) == T, "key_padding_mask time dim mismatch");

    // Check weight shapes: in_proj_weight (3*E, E), out_proj_weight (E, E)
    TORCH_CHECK(in_proj_weight.dim() == 2 && in_proj_weight.size(0) == 3 * E && in_proj_weight.size(1) == E,
                "in_proj_weight shape must be (3*embed_dim, embed_dim)");
    TORCH_CHECK(out_proj_weight.dim() == 2 && out_proj_weight.size(0) == E && out_proj_weight.size(1) == E,
                "out_proj_weight shape must be (embed_dim, embed_dim)");
    if (use_bias) {
        TORCH_CHECK(in_proj_bias_opt.value().dim() == 1 && in_proj_bias_opt.value().size(0) == 3 * E,
                    "in_proj_bias shape must be (3*embed_dim)");
        TORCH_CHECK(out_proj_bias_opt.value().dim() == 1 && out_proj_bias_opt.value().size(0) == E,
                    "out_proj_bias shape must be (embed_dim)");
    }

    // Allocate output tensor (same shape as query, i.e., (B, T, E) batch_first or (T, B, E) otherwise)
    auto output = torch::empty_like(query);

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Prepare pointers
    const float* query_ptr = query.data_ptr<float>();
    const float* key_ptr = key.data_ptr<float>();
    const float* value_ptr = value.data_ptr<float>();
    const float* in_proj_weight_ptr = in_proj_weight.data_ptr<float>();
    const float* in_proj_bias_ptr = use_bias ? in_proj_bias_opt.value().data_ptr<float>() : nullptr;
    const float* out_proj_weight_ptr = out_proj_weight.data_ptr<float>();
    const float* out_proj_bias_ptr = use_bias ? out_proj_bias_opt.value().data_ptr<float>() : nullptr;
    const bool* mask_ptr = key_padding_mask.data_ptr<bool>();
    float* output_ptr = output.data_ptr<float>();

    // Call launcher
    multihead_attention_forward_launcher(
        query_ptr, key_ptr, value_ptr,
        in_proj_weight_ptr, in_proj_bias_ptr,
        out_proj_weight_ptr, out_proj_bias_ptr,
        mask_ptr, output_ptr,
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(E), static_cast<int>(num_heads),
        batch_first, use_bias, true,  // mask_present always true because we require mask in model
        stream
    );

    return output;
}

// Elementwise multiply wrapper (using PyTorch's built-in multiply for now)
torch::Tensor multiply(torch::Tensor a, torch::Tensor b) {
    return a * b;
}

// Stack wrapper with explicit argument names
torch::Tensor stack(std::vector<torch::Tensor> tensors, int64_t dim) {
    return torch::stack(tensors, dim);
}

// Registration function
void register_multihead_attention(pybind11::module& m) {
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multihead attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("in_proj_weight"), py::arg("in_proj_bias"), py::arg("out_proj_weight"),
          py::arg("out_proj_bias"), py::arg("key_padding_mask"),
          py::arg("embed_dim"), py::arg("num_heads"), py::arg("batch_first"));
    m.def("multiply", &multiply, "Elementwise multiply", py::arg("a"), py::arg("b"));
    m.def("stack", &stack, "Stack tensors along dim", py::arg("tensors"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(multihead_attention, register_multihead_attention);
