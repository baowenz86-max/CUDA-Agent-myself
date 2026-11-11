#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void add_parameter_launcher(
    const float* x,
    const float* parameter,
    float* out,
    int64_t total,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t stride_x0,
    int64_t stride_x1,
    int64_t stride_x2,
    int64_t stride_x3,
    int64_t stride_param0,
    int64_t stride_param1,
    cudaStream_t stream);

// PyTorch wrapper for elementwise addition with broadcasting
torch::Tensor add_parameter(torch::Tensor x, torch::Tensor parameter) {
    // Validate device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(parameter.is_cuda(), "parameter must be a CUDA tensor");

    // Validate dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(parameter.dtype() == torch::kFloat32, "parameter must be float32");

    // Validate shapes: x must be 4D (as per contract), parameter must be [dim0, dim1, 1, 1]
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor");
    TORCH_CHECK(parameter.dim() == 4, "parameter must be a 4D tensor");
    TORCH_CHECK(parameter.size(0) == x.size(0) && parameter.size(1) == x.size(1),
                "parameter dim0/dim1 must match x");
    TORCH_CHECK(parameter.size(2) == 1 && parameter.size(3) == 1,
                "parameter must have shape [dim0, dim1, 1, 1]");

    // parameter must be contiguous (required by launcher's stride assumption)
    TORCH_CHECK(parameter.is_contiguous(), "parameter must be contiguous");

    // x may be non-contiguous; launcher handles arbitrary strides.
    // No need to enforce contiguity on x.

    // Allocate output with same shape, dtype, device as x
    auto output = torch::empty_like(x);

    // Get dimensions and strides
    int64_t dim0 = x.size(0);
    int64_t dim1 = x.size(1);
    int64_t dim2 = x.size(2);
    int64_t dim3 = x.size(3);
    int64_t total = x.numel();

    int64_t stride_x0 = x.stride(0);
    int64_t stride_x1 = x.stride(1);
    int64_t stride_x2 = x.stride(2);
    int64_t stride_x3 = x.stride(3);

    // parameter is contiguous, so strides: [dim1*1*1, 1*1, 1, 1] effectively [dim1, 1, 1, 1]
    // But we only need first two strides for indexing [i0, i1] in launcher.
    int64_t stride_param0 = parameter.stride(0); // should be dim1
    int64_t stride_param1 = parameter.stride(1); // should be 1

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    add_parameter_launcher(
        x.data_ptr<float>(),
        parameter.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_param0, stride_param1,
        stream);

    // Check for kernel launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_add_parameter(pybind11::module& m) {
    m.def("add_parameter", &add_parameter,
          "Elementwise addition with broadcasting: output = x + parameter",
          pybind11::arg("x"),
          pybind11::arg("parameter"));
}

// Auto-register using the provided macro
REGISTER_BINDING(add_parameter, register_add_parameter);
