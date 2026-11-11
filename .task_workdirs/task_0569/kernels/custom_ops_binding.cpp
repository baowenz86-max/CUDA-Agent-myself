#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void custom_abs_flat_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

// PyTorch wrapper: fused abs + flatten
// Input: 4D CUDA float32 tensor [N, C, H, W] (supports strides, but must be accessible contiguously
//        in underlying memory; we treat input as 3D collapsed after batch dimension)
// Output: 2D CUDA float32 tensor [N, C*H*W], contiguous
// The flatten operation (start_dim=1, end_dim=-1) merges dimensions after the first.
// For a contiguous input, the output is simply a contiguous copy of the absolute values
// in the same linear memory order. For non-contiguous inputs, we still operate on the
// underlying element order, which is equivalent to flattening the tensor as a 3D tensor
// [N, numel_per_batch] while preserving the batch size. Since the reference code uses
// torch.abs which preserves layout, and flatten on a contiguous tensor returns a contiguous
// view, we require contiguity for simplicity. If input is non-contiguous, we make it
// contiguous before launch (the reference model's input from get_inputs is always contiguous;
// explicit check avoids silently producing wrong results for strided inputs).
torch::Tensor custom_abs_flat_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs_flat_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs_flat_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_abs_flat_forward: input must be 4D [N, C, H, W]");

    // Make contiguous if needed. The reference model input is contiguous, but this check
    // ensures correctness for strided inputs as allowed by the contract (supports strides).
    auto x_contig = x.contiguous();

    int64_t batch = x_contig.size(0);
    int64_t numel_per_batch = x_contig.numel() / batch;
    int64_t total_numel = x_contig.numel();

    // Allocate output [batch, numel_per_batch] contiguous CUDA tensor
    auto output = torch::empty({batch, numel_per_batch}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch fused kernel (abs + linear copy; flatten is implicit via output shape)
    custom_abs_flat_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_numel,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs_flat_forward: CUDA kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_abs_flat_forward(pybind11::module& m) {
    m.def("custom_abs_flat_forward", &custom_abs_flat_forward,
          "Fused abs + flatten (start_dim=1, end_dim=-1)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_abs_flat_forward, register_custom_abs_flat_forward);
