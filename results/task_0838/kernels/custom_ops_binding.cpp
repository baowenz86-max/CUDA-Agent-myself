#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_acos_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

extern "C" void custom_reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int padding,
    cudaStream_t stream);

// Wrapper for custom_acos
torch::Tensor custom_acos(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_acos_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );

    return output;
}

// Wrapper for custom_reflection_pad2d
torch::Tensor custom_reflection_pad2d(torch::Tensor x, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    TORCH_CHECK(padding >= 0, "Padding must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t H_out = H + 2 * padding;
    int64_t W_out = W + 2 * padding;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(padding),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_acos", &custom_acos,
          "Custom acos operator",
          py::arg("x"));
    m.def("custom_reflection_pad2d", &custom_reflection_pad2d,
          "Custom reflection pad 2d operator",
          py::arg("x"),
          py::arg("padding"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
