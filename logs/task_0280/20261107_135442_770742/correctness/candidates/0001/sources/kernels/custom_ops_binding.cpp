#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int padding,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    cudaStream_t stream);

extern "C" void log2_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    cudaStream_t stream);

namespace {

torch::Tensor reflection_pad2d_forward(torch::Tensor x, int64_t padding) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t H_out = H_in + 2 * padding;
    int64_t W_out = W_in + 2 * padding;

    // Allocate contiguous output
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with input strides (may be non-contiguous)
    reflection_pad2d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H_in),
        static_cast<int>(W_in),
        static_cast<int>(padding),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return output;
}

torch::Tensor log2_forward(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate contiguous output
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with input strides
    log2_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          pybind11::arg("x"), pybind11::arg("padding"));
    m.def("log2_forward", &log2_forward,
          "Log2 elementwise forward",
          pybind11::arg("x"));
}

} // anonymous namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
