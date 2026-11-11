// Binding file for custom_ops.cu
// Contains PyTorch tensor handling and pybind registration only.
// No CUDA kernel implementations.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void triu_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t diagonal,
    int64_t input_batch_stride,
    int64_t input_channel_stride,
    int64_t input_row_stride,
    int64_t input_col_stride,
    int64_t output_batch_stride,
    int64_t output_channel_stride,
    int64_t output_row_stride,
    int64_t output_col_stride,
    cudaStream_t stream);

extern "C" void sum_dim23_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t input_batch_stride,
    int64_t input_channel_stride,
    int64_t input_row_stride,
    int64_t input_col_stride,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static inline void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error after launch: ", cudaGetErrorString(err));
}

// triu_forward: applies torch.triu(x, diagonal=diagonal)
torch::Tensor triu_forward(torch::Tensor x, int64_t diagonal) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "triu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "triu_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "triu_forward: input must have 4 dimensions");

    // Make sure input is contiguous for simplicity; launcher supports strides but
    // converting to contiguous ensures correct behavior even if input is non-contiguous.
    // However, per contract, layout supports strides, and reference input is contiguous.
    // We will handle general strides directly.
    auto x_contig = x.contiguous();

    int64_t batch_size = x_contig.size(0);
    int64_t channels = x_contig.size(1);
    int64_t height = x_contig.size(2);
    int64_t width = x_contig.size(3);

    // Strides are obtained from contiguous tensor, should be in elements.
    int64_t input_batch_stride = x_contig.stride(0);
    int64_t input_channel_stride = x_contig.stride(1);
    int64_t input_row_stride = x_contig.stride(2);
    int64_t input_col_stride = x_contig.stride(3);

    // Output tensor same shape, same options
    auto output = torch::empty_like(x_contig);

    // Output strides same as input (contiguous)
    int64_t output_batch_stride = output.stride(0);
    int64_t output_channel_stride = output.stride(1);
    int64_t output_row_stride = output.stride(2);
    int64_t output_col_stride = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    triu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        diagonal,
        input_batch_stride,
        input_channel_stride,
        input_row_stride,
        input_col_stride,
        output_batch_stride,
        output_channel_stride,
        output_row_stride,
        output_col_stride,
        stream
    );

    // Check for kernel errors
    check_cuda_error("triu_forward");

    return output;
}

// sum_dim23_forward: applies torch.sum(x, dim=(2,3))
torch::Tensor sum_dim23_forward(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "sum_dim23_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim23_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim23_forward: input must have 4 dimensions");

    // Make contiguous
    auto x_contig = x.contiguous();

    int64_t batch_size = x_contig.size(0);
    int64_t channels = x_contig.size(1);
    int64_t height = x_contig.size(2);
    int64_t width = x_contig.size(3);

    // Strides
    int64_t input_batch_stride = x_contig.stride(0);
    int64_t input_channel_stride = x_contig.stride(1);
    int64_t input_row_stride = x_contig.stride(2);
    int64_t input_col_stride = x_contig.stride(3);

    // Output shape [batch, channels]
    auto output = torch::empty({batch_size, channels}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sum_dim23_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        input_batch_stride,
        input_channel_stride,
        input_row_stride,
        input_col_stride,
        stream
    );

    // Check for kernel errors
    check_cuda_error("sum_dim23_forward");

    return output;
}

// Register both operators
void register_custom_ops(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Apply upper triangular mask (torch.triu)",
          py::arg("x"), py::arg("diagonal"));
    m.def("sum_dim23_forward", &sum_dim23_forward, "Sum over dimensions 2 and 3",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
