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
    cudaStream_t stream
);

// Custom embedding forward
// indices: CPU int64 tensor [batch, seq_len]
// weight: CPU float32 tensor [vocab_size, embedding_dim]
// vocab_size: int
// embedding_dim: int
// Returns: CUDA float32 tensor [batch, seq_len, embedding_dim]
torch::Tensor custom_embedding(
    torch::Tensor indices,
    torch::Tensor weight,
    int64_t vocab_size,
    int64_t embedding_dim
) {
    // Validate indices
    TORCH_CHECK(indices.is_cpu(), "indices must be a CPU tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must have dtype int64");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D [batch, seq_len]");
    // Validate weight
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must have dtype float32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [vocab_size, embedding_dim]");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight size(0) must equal vocab_size");
    TORCH_CHECK(weight.size(1) == embedding_dim, "weight size(1) must equal embedding_dim");

    // Make contiguous if needed (CPU tensors are often contiguous, but check)
    auto indices_contig = indices.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t batch = indices_contig.size(0);
    int64_t seq_len = indices_contig.size(1);

    // Allocate output CUDA float32 tensor [batch, seq_len, embedding_dim]
    auto output = torch::empty({batch, seq_len, embedding_dim},
                               indices_contig.options().dtype(torch::kFloat32).device(torch::kCUDA));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_embedding_launcher(
        indices_contig.data_ptr<int64_t>(),
        weight_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        seq_len,
        vocab_size,
        embedding_dim,
        stream
    );

    return output;
}

// Custom layer norm forward
// input: CUDA float32 tensor [batch, seq_len, normalized_dim]
// weight: optional CPU float32 tensor [normalized_dim]
// bias: optional CPU float32 tensor [normalized_dim]
// eps: double
// normalized_dim: int
// elementwise_affine: bool
// Returns: CUDA float32 tensor same shape as input
torch::Tensor custom_layer_norm(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps,
    int64_t normalized_dim,
    bool elementwise_affine
) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must have dtype float32");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(input.size(-1) == normalized_dim, "last dimension size must equal normalized_dim");

    // Ensure input contiguous (if not, make contiguous)
    auto input_contig = input.contiguous();

    int64_t rows = input_contig.numel() / normalized_dim;

    // Prepare optional CPU tensors
    torch::Tensor weight_contig;
    torch::Tensor bias_contig;
    const float* d_weight_ptr = nullptr;
    const float* d_bias_ptr = nullptr;

    if (elementwise_affine) {
        if (weight.has_value() && weight->defined()) {
            TORCH_CHECK(weight->is_cpu(), "weight must be a CPU tensor");
            TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must have dtype float32");
            TORCH_CHECK(weight->dim() == 1 && weight->numel() == normalized_dim,
                        "weight must be 1D with size normalized_dim");
            weight_contig = weight->contiguous();
            d_weight_ptr = weight_contig.data_ptr<float>();
        }
        if (bias.has_value() && bias->defined()) {
            TORCH_CHECK(bias->is_cpu(), "bias must be a CPU tensor");
            TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must have dtype float32");
            TORCH_CHECK(bias->dim() == 1 && bias->numel() == normalized_dim,
                        "bias must be 1D with size normalized_dim");
            bias_contig = bias->contiguous();
            d_bias_ptr = bias_contig.data_ptr<float>();
        }
    }

    // Allocate output CUDA float32 tensor same shape as input
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_layer_norm_launcher(
        input_contig.data_ptr<float>(),
        d_weight_ptr,
        d_bias_ptr,
        output.data_ptr<float>(),
        rows,
        normalized_dim,
        eps,
        elementwise_affine,
        stream
    );

    return output;
}

// Register Python bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding", &custom_embedding,
          "Custom embedding forward",
          py::arg("indices"),
          py::arg("weight"),
          py::arg("vocab_size"),
          py::arg("embedding_dim"));

    m.def("custom_layer_norm", &custom_layer_norm,
          "Custom layer norm forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"),
          py::arg("normalized_dim"),
          py::arg("elementwise_affine"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
