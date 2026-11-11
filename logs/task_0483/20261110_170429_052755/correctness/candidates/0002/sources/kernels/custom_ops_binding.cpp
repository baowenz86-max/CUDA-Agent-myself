// Binding for adaptive_avg_pool1d_transpose custom CUDA kernel.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu (C++ linkage).
void adaptive_avg_pool1d_transpose_launcher(
    float* output,
    const float* input,
    int B, int C, int W, int output_size,
    cudaStream_t stream);

// PyTorch wrapper: input [B, C, W] CUDA float32 tensor, output_size int.
// Returns tensor with shape [B, W, output_size] and dtype float32, contiguous.
torch::Tensor adaptive_avg_pool1d_transpose(
    torch::Tensor x,
    int64_t output_size)
{
    // Validate device and dtype.
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3-dimensional [B, C, W]");

    // We require contiguous input for the kernel's layout assumption.
    // The contract says 'layout: supports strides', so we call .contiguous() to support strided inputs safely.
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int64_t B = sizes[0];
    int64_t C = sizes[1];
    int64_t W = sizes[2];

    TORCH_CHECK(output_size > 0, "output_size must be positive");
    TORCH_CHECK(output_size <= C, "output_size cannot exceed channel dimension");

    // Allocate output with shape [B, W, output_size] contiguous.
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x_contig.device());
    torch::Tensor output = torch::empty({B, W, output_size}, options);

    // Get current CUDA stream.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch CUDA kernel.
    adaptive_avg_pool1d_transpose_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(W),
        static_cast<int>(output_size),
        stream);

    // Check for launch errors.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register the Python function.
void register_adaptive_avg_pool1d_transpose(pybind11::module& m) {
    m.def(
        "adaptive_avg_pool1d_transpose",
        &adaptive_avg_pool1d_transpose,
        "Adaptive average pooling over channel dimension after transposing input from [B,C,W] to [B,W,output_size]",
        py::arg("x"),
        py::arg("output_size"));
}

REGISTER_BINDING(adaptive_avg_pool1d_transpose, register_adaptive_avg_pool1d_transpose);
