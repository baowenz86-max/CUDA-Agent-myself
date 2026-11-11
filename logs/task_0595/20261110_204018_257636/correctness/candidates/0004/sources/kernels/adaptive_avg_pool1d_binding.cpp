#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare the CUDA launcher from custom_ops.cu
extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    long long batch,
    long long channels,
    long long seq_len,
    long long output_size,
    cudaStream_t stream);

torch::Tensor adaptive_avg_pool1d_forward(torch::Tensor input, long output_size) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch, channels, seq_len)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    // Make input contiguous for the kernel (the kernel expects row-major per (b,c))
    auto input_contig = input.contiguous();

    int64_t batch = input_contig.size(0);
    int64_t channels = input_contig.size(1);
    int64_t seq_len = input_contig.size(2);

    // Allocate output tensor with shape [batch, channels, output_size]
    auto output = torch::empty({batch, channels, static_cast<int64_t>(output_size)}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_avg_pool1d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(batch),
        static_cast<long long>(channels),
        static_cast<long long>(seq_len),
        static_cast<long long>(output_size),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_avg_pool1d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_adaptive_avg_pool1d(py::module& m) {
    m.def(
        "adaptive_avg_pool1d",
        &adaptive_avg_pool1d_forward,
        "Adaptive average pooling over last dimension",
        py::arg("input"),
        py::arg("output_size")
    );
}

REGISTER_BINDING(adaptive_avg_pool1d, register_adaptive_avg_pool1d);