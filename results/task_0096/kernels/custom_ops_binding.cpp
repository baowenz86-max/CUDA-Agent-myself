#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void softmax_dim1_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    cudaStream_t stream
);

// PyTorch wrapper for softmax over dim=1
torch::Tensor softmax_dim1_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch, channels, length)");

    // Ensure input is contiguous; if not, make contiguous copy
    auto input_contig = input.contiguous();

    int batch = input_contig.size(0);
    int channels = input_contig.size(1);
    int length = input_contig.size(2);

    // Allocate output with same shape and options
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    softmax_dim1_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        batch,
        channels,
        length,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_dim1_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_softmax_dim1(pybind11::module& m) {
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax over dim=1 for 3D tensor");
}

REGISTER_BINDING(softmax_dim1, register_softmax_dim1);
