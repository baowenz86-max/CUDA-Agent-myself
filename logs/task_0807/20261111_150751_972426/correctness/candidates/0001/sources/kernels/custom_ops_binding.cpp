#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void fused_relu_cos_scale_launcher(
    float* output,
    const float* input,
    int64_t numel,
    double scale,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t input_stride_0,
    int64_t input_stride_1,
    int64_t input_stride_2,
    int64_t input_stride_3,
    cudaStream_t stream);

// PyTorch wrapper for fused_relu_cos_scale
// x: input tensor (float32, CUDA), arbitrary strides allowed
// scale: double scalar (represents out_features in the model)
// returns: new contiguous tensor with same shape as x
torch::Tensor fused_relu_cos_scale(torch::Tensor x, double scale) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional");

    // Allocate output contiguous tensor of same shape
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract dimensions
    int64_t dim0 = x.size(0);
    int64_t dim1 = x.size(1);
    int64_t dim2 = x.size(2);
    int64_t dim3 = x.size(3);

    // Get input strides (in elements)
    int64_t input_stride_0 = x.stride(0);
    int64_t input_stride_1 = x.stride(1);
    int64_t input_stride_2 = x.stride(2);
    int64_t input_stride_3 = x.stride(3);

    // Number of elements
    int64_t numel = x.numel();

    // Call launcher
    fused_relu_cos_scale_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        scale,
        dim0,
        dim1,
        dim2,
        dim3,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        input_stride_3,
        stream);

    return output;
}

// Registration
void register_fused_relu_cos_scale(pybind11::module& m) {
    m.def(
        "fused_relu_cos_scale",
        &fused_relu_cos_scale,
        "Fused ReLU + cos + scale operation on CUDA float32 tensor",
        py::arg("x"),
        py::arg("scale"));
}

REGISTER_BINDING(fused_relu_cos_scale, register_fused_relu_cos_scale);
