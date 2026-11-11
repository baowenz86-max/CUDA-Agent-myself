// Binding for custom softsign CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void softsign_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    cudaStream_t stream);

// PyTorch wrapper: elementwise softsign, preserving shape, dtype, device, strides
torch::Tensor softsign_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");

    // Output tensor: same shape, dtype, device; use empty_like to preserve strides
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract dimensions and strides as int64_t
    int64_t total_elements = x.numel();
    int64_t dims[4];
    int64_t strides[4];
    for (int i = 0; i < 4; ++i) {
        dims[i] = x.size(i);
        strides[i] = x.stride(i);
    }

    // Call launcher with raw pointers
    softsign_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        dims[0], dims[1], dims[2], dims[3],
        strides[0], strides[1], strides[2], strides[3],
        stream
    );

    return output;
}

// Registration function
void register_softsign(pybind11::module& m) {
    m.def("softsign_forward", &softsign_forward,
          "Elementwise softsign forward (CUDA)",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(softsign, register_softsign);
