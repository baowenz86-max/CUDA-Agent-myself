#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void flip_forward_launcher(
    float* output,
    const float* input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t ndim,
    int64_t total_elements,
    cudaStream_t stream);

torch::Tensor flip_forward(torch::Tensor x, int64_t dim) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() > dim && dim >= -x.dim(), "dim out of range");

    // Convert negative dim to positive
    if (dim < 0) {
        dim += x.dim();
    }

    // Allocate output tensor with same shape and options
    auto output = torch::empty(x.sizes(), x.options());

    int64_t ndim = x.dim();
    TORCH_CHECK(ndim <= 4, "Only up to 4 dimensions supported");

    // Fill sizes and strides with padding for unused dimensions
    int64_t sizes[4] = {1, 1, 1, 1};
    int64_t strides[4] = {0, 0, 0, 0};
    for (int64_t i = 0; i < ndim; ++i) {
        sizes[i] = x.size(i);
        strides[i] = x.stride(i);
    }

    int64_t total_elements = x.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    flip_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        dim,
        sizes[0], sizes[1], sizes[2], sizes[3],
        strides[0], strides[1], strides[2], strides[3],
        ndim,
        total_elements,
        stream);

    return output;
}

void register_flip_forward(pybind11::module& m) {
    m.def("flip_forward", &flip_forward,
          "Flip tensor along a single dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(flip_forward, register_flip_forward);
