// Binding for scaled_dot_product_attention_forward
// Only float32, contiguous, CUDA tensors supported.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from sdpa_kernel.cu
extern "C" void sdpa_forward_launcher(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* attn_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);

torch::Tensor scaled_dot_product_attention_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor attn_mask) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(attn_mask.is_cuda(), "attn_mask must be a CUDA tensor");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(value.dtype() == torch::kFloat32, "value must be float32");
    TORCH_CHECK(attn_mask.dtype() == torch::kFloat32, "attn_mask must be float32");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(attn_mask.is_contiguous(), "attn_mask must be contiguous");
    TORCH_CHECK(query.dim() == 3, "query must be 3D (batch, seq_len, head_dim)");
    TORCH_CHECK(key.dim() == 3, "key must be 3D (batch, seq_len, head_dim)");
    TORCH_CHECK(value.dim() == 3, "value must be 3D (batch, seq_len, head_dim)");
    TORCH_CHECK(attn_mask.dim() == 2, "attn_mask must be 2D (seq_len, seq_len)");
    TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0), "batch size mismatch");
    TORCH_CHECK(query.size(1) == key.size(1) && query.size(1) == value.size(1), "seq_len mismatch");
    TORCH_CHECK(query.size(2) == key.size(2) && query.size(2) == value.size(2), "head_dim mismatch");
    TORCH_CHECK(attn_mask.size(0) == query.size(1) && attn_mask.size(1) == query.size(1), "attn_mask shape mismatch");

    int batch_size = query.size(0);
    int seq_len = query.size(1);
    int head_dim = query.size(2);

    auto output = torch::empty_like(query);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sdpa_forward_launcher(
        output.data_ptr<float>(),
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        attn_mask.data_ptr<float>(),
        batch_size, seq_len, head_dim,
        stream
    );

    return output;
}

void register_sdpa(py::module& m) {
    m.def("scaled_dot_product_attention_forward", &scaled_dot_product_attention_forward,
          "Scaled dot product attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("attn_mask"));
}

REGISTER_BINDING(sdpa, register_sdpa);
