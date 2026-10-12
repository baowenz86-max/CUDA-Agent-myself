// Binding for baddbmm_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from baddbmm_kernel.cu
extern "C" void baddbmm_launcher(float* output, const float* input, const float* batch1,
                                 const float* batch2, float beta, float alpha, int B, int N,
                                 int M, int P, cudaStream_t stream);

torch::Tensor baddbmm_forward(torch::Tensor input, torch::Tensor batch1, torch::Tensor batch2) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(batch1.is_cuda(), "batch1 must be a CUDA tensor");
    TORCH_CHECK(batch2.is_cuda(), "batch2 must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(batch1.dtype() == torch::kFloat32, "batch1 must be float32");
    TORCH_CHECK(batch2.dtype() == torch::kFloat32, "batch2 must be float32");

    // Ensure contiguous tensors (required by the custom kernel)
    auto batch1_contig = batch1.contiguous();
    auto batch2_contig = batch2.contiguous();
    // input is only used if beta != 0, and we always set beta=0 so it doesn't need to be contiguous.
    // However, for safety, we still pass its data pointer; ensure it's allocated.
    // input is zeros_like(target) and contiguous from model_new.py, but we can make it contiguous too.
    auto input_contig = input.contiguous();

    // Get dimensions
    auto b1_sizes = batch1_contig.sizes();
    auto b2_sizes = batch2_contig.sizes();
    TORCH_CHECK(b1_sizes.size() == 3, "batch1 must be 3D (B, N, M)");
    TORCH_CHECK(b2_sizes.size() == 3, "batch2 must be 3D (B, M, P)");

    int B = b1_sizes[0];
    int N = b1_sizes[1];
    int M = b1_sizes[2];
    int B2 = b2_sizes[0];
    int P = b2_sizes[1];
    int M2 = b2_sizes[2];

    TORCH_CHECK(B == B2, "batch1 and batch2 must have same batch size");
    TORCH_CHECK(M == M2, "inner dimensions must match for matrix multiplication");

    // Output shape: [B, N, P]
    auto output = torch::empty({B, N, P}, batch1_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with beta=0, alpha=1 (we replicate torch.baddbmm with beta=0)
    baddbmm_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        batch1_contig.data_ptr<float>(),
        batch2_contig.data_ptr<float>(),
        0.0f, 1.0f,
        B, N, M, P,
        stream
    );

    return output;
}

void register_baddbmm(pybind11::module& m) {
    m.def("baddbmm_forward", &baddbmm_forward, "Batched matrix multiplication with beta=0, alpha=1");
}

REGISTER_BINDING(baddbmm, register_baddbmm);
