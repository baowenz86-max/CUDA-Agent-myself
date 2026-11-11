#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_forward_launcher(
    const float* x1,
    const float* x2,
    float* out,
    long long x1_numel,
    long long x2_numel,
    cudaStream_t stream);

extern "C" void custom_ceil_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

static torch::Tensor custom_cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "x2 must be contiguous");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported for cat");

    auto x1_numel = x1.numel();
    auto x2_numel = x2.numel();

    // Compute output shape: [x1.size(0) + x2.size(0), ...]
    std::vector<int64_t> out_sizes(x1.sizes().begin(), x1.sizes().end());
    out_sizes[0] = x1.size(0) + x2.size(0);

    auto out = torch::empty(out_sizes, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_cat_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<long long>(x1_numel),
        static_cast<long long>(x2_numel),
        stream);

    return out;
}

static torch::Tensor custom_ceil_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_ceil_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream);

    return out;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat_forward", &custom_cat_forward,
          "Concatenate two tensors along dim=0",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("custom_ceil_forward", &custom_ceil_forward,
          "Element-wise ceil",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
