// kernels/custom_ops_binding.cpp
// PyTorch tensor wrappers and pybind registrations for custom_ops.cu launchers.
// Conv/ConvTranspose are not relevant in this task; no cuDNN required.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <chrono>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_embedding_forward_launcher(
    const float* weight_ptr,
    const int64_t* indices_ptr,
    float* output_ptr,
    int embedding_dim,
    int64_t num_indices,
    int64_t weight_row_stride,
    int64_t out_row_stride,
    cudaStream_t stream);

extern "C" void custom_randn_like_launcher(
    float* output_ptr,
    int64_t total_numel,
    uint64_t seed,
    cudaStream_t stream);

extern "C" void custom_smooth_l1_loss_forward_launcher(
    const float* input_ptr,
    const float* target_ptr,
    float* output_ptr,
    int64_t total_numel,
    cudaStream_t stream);

// Helper: get current CUDA stream
static inline cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// -----------------------------------------------------------------------------
// custom_embedding_forward
// weight: CUDA float32 tensor [num_embeddings, embedding_dim]
// indices: CUDA int64 tensor of arbitrary shape
// returns: CUDA float32 tensor shape = indices.sizes() + [embedding_dim]
torch::Tensor custom_embedding_forward(torch::Tensor weight, torch::Tensor indices) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    auto weight_contig = weight.contiguous();
    int64_t embedding_dim = weight_contig.size(1);
    int64_t num_indices = indices.numel();

    // Construct output shape: indices.sizes() + [embedding_dim]
    std::vector<int64_t> out_sizes(indices.sizes().begin(), indices.sizes().end());
    out_sizes.push_back(embedding_dim);

    auto output = torch::empty(out_sizes, weight_contig.options());

    cudaStream_t stream = getCurrentStream();
    custom_embedding_forward_launcher(
        weight_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        static_cast<int>(embedding_dim),
        num_indices,
        embedding_dim,  // weight_row_stride (contiguous)
        embedding_dim,  // out_row_stride (contiguous, last dim = embedding_dim)
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// custom_randn_like
// input: CUDA float32 tensor of any shape
// returns: CUDA float32 tensor of same shape, filled with standard normal random numbers
torch::Tensor custom_randn_like(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int64_t total_numel = input.numel();

    // Seed: combine current time and input data pointer to get a varying seed.
    // This does not guarantee exact torch.randn_like distribution but is sufficient
    // for the contract (no exact value matching required).
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t seed = static_cast<uint64_t>(now) ^ reinterpret_cast<uint64_t>(input.data_ptr<float>());

    cudaStream_t stream = getCurrentStream();
    custom_randn_like_launcher(
        output.data_ptr<float>(),
        total_numel,
        seed,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// custom_smooth_l1_loss_forward
// input, target: CUDA float32 tensors of same shape
// returns: CUDA float32 scalar tensor (0-dim), mean of smooth l1 loss
torch::Tensor custom_smooth_l1_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have the same shape");
    TORCH_CHECK(input.numel() > 0, "input must have at least one element");

    auto output = torch::empty({}, input.options());
    cudaStream_t stream = getCurrentStream();
    custom_smooth_l1_loss_forward_launcher(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding_forward", &custom_embedding_forward,
          "Embedding forward (CUDA)",
          py::arg("weight"), py::arg("indices"));
    m.def("custom_randn_like", &custom_randn_like,
          "Generate random tensor like input (CUDA)",
          py::arg("input"));
    m.def("custom_smooth_l1_loss_forward", &custom_smooth_l1_loss_forward,
          "Smooth L1 loss forward (CUDA, mean reduction)",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
