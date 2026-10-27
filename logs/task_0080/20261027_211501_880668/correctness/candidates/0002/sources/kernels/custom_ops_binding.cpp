#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void prelu_forward_launcher(
    const float* input,
    float* output,
    float weight,
    int64_t numel,
    cudaStream_t stream);

extern "C" void masked_select_count_launcher(
    const float* input,
    const bool* mask,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta,
    int64_t* block_counts,
    int64_t* block_offsets,
    int64_t* total_count,
    int num_blocks,
    cudaStream_t stream);

extern "C" void masked_select_gather_launcher(
    const float* input,
    const bool* mask,
    float* output,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta,
    const int64_t* block_offsets,
    int num_blocks,
    cudaStream_t stream);

// Define TensorMeta (must match .cu definition)
struct TensorMeta {
    int ndim;
    int64_t shape[4];
    int64_t strides[4];
};

// Helper: fill TensorMeta from a torch::Tensor
TensorMeta get_meta(const torch::Tensor& t) {
    TensorMeta meta;
    meta.ndim = t.dim();
    TORCH_CHECK(meta.ndim <= 4, "Only up to 4D tensors are supported");
    for (int i = 0; i < 4; ++i) {
        meta.shape[i] = 0;
        meta.strides[i] = 0;
    }
    for (int i = 0; i < meta.ndim; ++i) {
        meta.shape[i] = t.size(i);
        meta.strides[i] = t.stride(i);
    }
    return meta;
}

// masked_select_forward
// Returns 1D tensor of selected values in row-major order.
torch::Tensor masked_select_forward(torch::Tensor input, torch::Tensor mask) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");

    // Broadcast check: mask shape must be broadcastable to input shape.
    // PyTorch broadcasting rules: trailing dims must be equal or one of them is 1.
    int input_ndim = input.dim();
    int mask_ndim = mask.dim();
    TORCH_CHECK(mask_ndim <= input_ndim, "mask dims cannot exceed input dims");
    for (int i = 1; i <= mask_ndim; ++i) {
        int64_t in_dim = input.size(input_ndim - i);
        int64_t mask_dim = mask.size(mask_ndim - i);
        TORCH_CHECK(in_dim == mask_dim || in_dim == 1 || mask_dim == 1,
                    "mask shape not broadcastable to input shape");
    }

    // Ensure input and mask are contiguous for kernel simplicity?
    // The .cu kernels assume linear index over input and use strides to compute mask offset,
    // but they access input[idx] directly, which requires input contiguous. We'll make contiguous copies.
    auto input_contig = input.contiguous();
    auto mask_contig = mask.contiguous();

    int64_t total_elements = input_contig.numel();
    TensorMeta input_meta = get_meta(input_contig);
    TensorMeta mask_meta = get_meta(mask_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Choose block size and items per thread to match .cu kernels
    const int threads = 256;
    constexpr int ITEMS_PER_THREAD = 8;
    int64_t num_blocks = (total_elements + (int64_t)threads * ITEMS_PER_THREAD - 1) /
                         ((int64_t)threads * ITEMS_PER_THREAD);
    TORCH_CHECK(num_blocks > 0, "input must not be empty");

    // Allocate block_counts and block_offsets on device
    auto block_counts = torch::empty({num_blocks},
                                     torch::dtype(torch::kInt64).device(input.device()));
    auto block_offsets = torch::empty({num_blocks},
                                      torch::dtype(torch::kInt64).device(input.device()));
    auto total_count_tensor = torch::empty({1},
                                           torch::dtype(torch::kInt64).device(input.device()));

    // First launcher: count and scan
    masked_select_count_launcher(
        input_contig.data_ptr<float>(),
        mask_contig.data_ptr<bool>(),
        total_elements,
        input_meta,
        mask_meta,
        block_counts.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        total_count_tensor.data_ptr<int64_t>(),
        (int)num_blocks,
        stream);

    // Copy total count to CPU
    int64_t total_count_cpu;
    cudaMemcpyAsync(&total_count_cpu, total_count_tensor.data_ptr<int64_t>(),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // ensure copy complete

    // Allocate output
    auto output = torch::empty({total_count_cpu},
                               torch::dtype(torch::kFloat32).device(input.device()));

    // Second launcher: gather
    masked_select_gather_launcher(
        input_contig.data_ptr<float>(),
        mask_contig.data_ptr<bool>(),
        output.data_ptr<float>(),
        total_elements,
        input_meta,
        mask_meta,
        block_offsets.data_ptr<int64_t>(),
        (int)num_blocks,
        stream);

    return output;
}

// prelu_forward
// Applies PReLU with a single weight (num_parameters=1).
torch::Tensor prelu_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.numel() == 1, "weight must have exactly one element");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy weight to CPU to pass as scalar (simple, one element)
    float weight_cpu;
    cudaMemcpyAsync(&weight_cpu, weight.data_ptr<float>(), sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    prelu_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_cpu,
        input_contig.numel(),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("masked_select_forward", &masked_select_forward,
          "Stable masked_select implementation (CUDA)",
          pybind11::arg("input"), pybind11::arg("mask"));
    m.def("prelu_forward", &prelu_forward,
          "PReLU with single weight (CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
