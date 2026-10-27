// Binding file for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim1_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    cudaStream_t stream
);

extern "C" void smooth_l1_loss_launcher(
    const float* input,
    const float* target,
    float* out,
    long long N,
    cudaStream_t stream
);

// Wrapper for mean_dim1
torch::Tensor mean_dim1(torch::Tensor x) {
    // Check CUDA
    TORCH_CHECK(x.is_cuda(), "mean_dim1: input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim1: input must be float32");
    // Check dims
    TORCH_CHECK(x.dim() == 4, "mean_dim1: input must be 4D (B,C,H,W)");

    // Extract dims
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output [B, H, W] contiguous
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor out = torch::empty({B, H, W}, options);

    // Get strides (as long long)
    long long sx_b = x.stride(0);
    long long sx_c = x.stride(1);
    long long sx_h = x.stride(2);
    long long sx_w = x.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with original pointers (non-contiguous handled via strides)
    mean_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        sx_b, sx_c, sx_h, sx_w,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for smooth_l1_loss
torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target) {
    // Check CUDA
    TORCH_CHECK(input.is_cuda(), "smooth_l1_loss: input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "smooth_l1_loss: target must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(input.dtype() == torch::kFloat32, "smooth_l1_loss: input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "smooth_l1_loss: target must be float32");
    // Check shapes exactly
    TORCH_CHECK(input.sizes() == target.sizes(), "smooth_l1_loss: input and target must have same shape");

    // Number of elements
    long long N = input.numel();
    TORCH_CHECK(N > 0, "smooth_l1_loss: input must not be empty");

    // Allocate output scalar tensor (shape [])
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor out = torch::empty({}, options);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Ensure input and target are contiguous for simple elementwise access.
    // The launcher assumes contiguous memory for both tensors.
    // We can make them contiguous if needed (no effect if already contiguous).
    torch::Tensor input_contig = input.contiguous();
    torch::Tensor target_contig = target.contiguous();

    // Call launcher
    smooth_l1_loss_launcher(
        input_contig.data_ptr<float>(),
        target_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "smooth_l1_loss kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim1", &mean_dim1, "Mean over dimension 1", py::arg("x"));
    m.def("smooth_l1_loss", &smooth_l1_loss, "Smooth L1 loss", py::arg("input"), py::arg("target"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
