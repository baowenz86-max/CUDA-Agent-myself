#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include "../binding_registry.h"

// Declare launchers exported by custom_ops.cu
extern "C" {
void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    long long sB,
    long long sC,
    long long sH,
    long long sW,
    cudaStream_t stream);

void prelu_launcher(
    const float* x,
    float* out,
    const float* weight,
    int total,
    cudaStream_t stream);
}

// Operator 1: sum_dim1_keepdim
// Input: x [B, C, H, W] float32 CUDA, may be non-contiguous.
// Output: [B, 1, H, W] float32 CUDA
torch::Tensor sum_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_dim1_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "sum_dim1_keepdim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim1_keepdim: input must be 4D [B, C, H, W]");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Use input strides converted to long long to match launcher signature.
    long long sB = static_cast<long long>(x.stride(0));
    long long sC = static_cast<long long>(x.stride(1));
    long long sH = static_cast<long long>(x.stride(2));
    long long sW = static_cast<long long>(x.stride(3));

    sum_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        sB, sC, sH, sW,
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Operator 2: prelu
// Input: x [B, 1, H, W] float32 CUDA, weight [1] float32 CUDA
// Output: same shape as x, elementwise PReLU
torch::Tensor prelu(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "prelu: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "prelu: weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "prelu: x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "prelu: weight must be float32");
    TORCH_CHECK(x.dim() == 4, "prelu: x must be 4D");
    TORCH_CHECK(weight.numel() == 1, "prelu: weight must have exactly one element");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int total = x.numel();
    prelu_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        total,
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void register_custom_ops(pybind11::module& m) {
    namespace py = pybind11;
    m.def("sum_dim1_keepdim", &sum_dim1_keepdim,
          py::arg("x"));
    m.def("prelu", &prelu,
          py::arg("x"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
