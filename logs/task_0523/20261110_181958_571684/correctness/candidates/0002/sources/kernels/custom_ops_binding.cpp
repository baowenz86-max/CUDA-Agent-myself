#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void flip_forward_launcher(
    float* output,
    const float* input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t total_elements,
    cudaStream_t stream);

torch::Tensor flip_forward(torch::Tensor x, int64_t dim) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "This kernel is specialized for 4D tensors");
    TORCH_CHECK(dim >= -x.dim() && dim < x.dim(), "dim out of range");

    // Convert negative dim to positive
    if (dim < 0) {
        dim += x.dim();
    }

    // Allocate output tensor with same shape and options
    auto output = torch::empty(x.sizes(), x.options());

    // Extract sizes and strides
    int64_t size0 = x.size(0);
    int64_t size1 = x.size(1);
    int64_t size2 = x.size(2);
    int64_t size3 = x.size(3);
    int64_t stride0 = x.stride(0);
    int64_t stride1 = x.stride(1);
    int64_t stride2 = x.stride(2);
    int64_t stride3 = x.stride(3);
    int64_t total_elements = x.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    flip_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        dim,
        size0, size1, size2, size3,
        stride0, stride1, stride2, stride3,
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
