#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t vocab_size,
    int64_t embedding_dim,
    bool indices_on_device,
    bool weight_on_device,
    cudaStream_t stream
);

extern "C" void custom_layer_norm_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t rows,
    int64_t dim,
    double eps,
    bool elementwise_affine,
    bool weight_on_device,
    bool bias_on_device,
    cudaStream_t stream
);

// custom_embedding forward
// indices: int64 tensor [batch, seq_len] (CPU or CUDA)
// weight: float32 tensor [vocab_size, embedding_dim] (CPU or CUDA)
// returns CUDA float32 tensor [batch, seq_len, embedding_dim]
torch::Tensor custom_embedding(
    torch::Tensor indices,
    torch::Tensor weight,
    int64_t vocab_size,
    int64_t embedding_dim
) {
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must have dtype int64");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch, seq_len]");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must have dtype float32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [vocab_size, embedding_dim]");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight size(0) must equal vocab_size");
    TORCH_CHECK(weight.size(1) == embedding_dim, "weight size(1) must equal embedding_dim");

    int64_t batch = indices.size(0);
    int64_t seq_len = indices.size(1);

    // Allocate output on CUDA
    auto output = torch::empty({batch, seq_len, embedding_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bool indices_on_device = indices.is_cuda();
    bool weight_on_device = weight.is_cuda();

    // Ensure contiguous; if CUDA, use contiguous directly; if CPU, also contiguous (host)
    torch::Tensor indices_contig = indices.contiguous();
    torch::Tensor weight_contig = weight.contiguous();

    const int64_t* indices_ptr = indices_contig.data_ptr<int64_t>();
    const float* weight_ptr = weight_contig.data_ptr<float>();

    custom_embedding_launcher(
        indices_ptr,
        weight_ptr,
        output.data_ptr<float>(),
        batch,
        seq_len,
        vocab_size,
        embedding_dim,
        indices_on_device,
        weight_on_device,
        stream
    );

    return output;
}

// custom_layer_norm forward
// input: CUDA float32 tensor [batch, seq_len, normalized_dim]
// weight: optional float32 tensor [normalized_dim] (CPU or CUDA, may be None)
// bias: optional float32 tensor [normalized_dim] (CPU or CUDA, may be None)
// returns CUDA float32 tensor same shape
torch::Tensor custom_layer_norm(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps,
    int64_t normalized_dim,
    bool elementwise_affine
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must have dtype float32");
    TORCH_CHECK(input.size(-1) == normalized_dim, "last dimension must match normalized_dim");

    auto input_contig = input.contiguous();
    int64_t rows = input_contig.numel() / normalized_dim;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor weight_contig;
    torch::Tensor bias_contig;
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    bool weight_on_device = false;
    bool bias_on_device = false;

    if (elementwise_affine) {
        if (weight.has_value() && weight->defined()) {
            TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
            TORCH_CHECK(weight->dim() == 1 && weight->numel() == normalized_dim, "weight size mismatch");
            weight_contig = weight->contiguous();
            weight_ptr = weight_contig.data_ptr<float>();
            weight_on_device = weight->is_cuda();
        }
        if (bias.has_value() && bias->defined()) {
            TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
            TORCH_CHECK(bias->dim() == 1 && bias->numel() == normalized_dim, "bias size mismatch");
            bias_contig = bias->contiguous();
            bias_ptr = bias_contig.data_ptr<float>();
            bias_on_device = bias->is_cuda();
        }
    }

    auto output = torch::empty_like(input_contig);

    custom_layer_norm_launcher(
        input_contig.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        output.data_ptr<float>(),
        rows,
        normalized_dim,
        eps,
        elementwise_affine,
        weight_on_device,
        bias_on_device,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding", &custom_embedding,
          "Custom embedding forward",
          py::arg("indices"), py::arg("weight"), py::arg("vocab_size"), py::arg("embedding_dim"));
    m.def("custom_layer_norm", &custom_layer_norm,
          "Custom layer norm forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps"),
          py::arg("normalized_dim"), py::arg("elementwise_affine"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
