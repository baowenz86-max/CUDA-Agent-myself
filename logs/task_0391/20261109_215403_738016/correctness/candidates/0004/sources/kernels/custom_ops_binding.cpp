#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void expand_mask_launcher(
    const bool* input,
    bool* output,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void masked_scatter_launcher(
    const float* input,
    const bool* mask,
    const float* source,
    float* output,
    int64_t total_elements,
    int64_t source_numel,
    int* d_block_totals,
    int* d_block_flags,
    cudaStream_t stream
);

// Wrapper for expand_mask
torch::Tensor expand_mask(torch::Tensor mask, int64_t size) {
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");

    // The mask is already expanded and contiguous. If not contiguous, make a copy.
    if (mask.is_contiguous()) {
        return mask;
    }

    auto output = torch::empty_like(mask);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    expand_mask_launcher(
        mask.data_ptr<bool>(),
        output.data_ptr<bool>(),
        mask.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "expand_mask launch failed: ", cudaGetErrorString(err));
    return output;
}

// Wrapper for masked_scatter
torch::Tensor masked_scatter(torch::Tensor input, torch::Tensor mask, torch::Tensor source) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
    TORCH_CHECK(source.dtype() == torch::kFloat32, "source must be float32");

    // Ensure contiguity; .contiguous() returns same tensor if already contiguous.
    auto input_contig = input.contiguous();
    auto mask_contig = mask.contiguous();
    auto source_contig = source.contiguous();

    TORCH_CHECK(input_contig.numel() == mask_contig.numel(), "input and mask must have same number of elements");

    int64_t total = input_contig.numel();
    int64_t source_numel = source_contig.numel();
    TORCH_CHECK(source_numel <= total, "source has more elements than mask positions");

    auto output = torch::empty_like(input_contig);

    const int BLOCK_SIZE = 256;
    int64_t num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto block_totals = torch::empty({num_blocks}, torch::dtype(torch::kInt32).device(input_contig.device()));
    auto block_flags = torch::empty({num_blocks}, torch::dtype(torch::kInt32).device(input_contig.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_scatter_launcher(
        input_contig.data_ptr<float>(),
        mask_contig.data_ptr<bool>(),
        source_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        source_numel,
        block_totals.data_ptr<int>(),
        block_flags.data_ptr<int>(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "masked_scatter launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("expand_mask", &expand_mask, "Expand mask to target shape",
          py::arg("mask"), py::arg("size") = 4);
    m.def("masked_scatter", &masked_scatter, "Masked scatter operation",
          py::arg("input"), py::arg("mask"), py::arg("source"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
