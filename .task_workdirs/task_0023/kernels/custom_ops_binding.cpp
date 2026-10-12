// Binding for median_dim0_forward and reflection_pad2d_forward custom kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void median_dim0_forward_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void reflection_pad2d_forward_launcher(
    const float* input, float* output,
    int C, int H, int W, int pad,
    cudaStream_t stream);

// Wrapper for median along dim0
torch::Tensor median_dim0_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Output tensor shape [C, H, W]
    auto output = torch::empty({C, H, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    median_dim0_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// Wrapper for reflection padding 2D
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t pad_size) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [C, H, W]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(pad_size >= 0, "pad_size must be non-negative");

    int C = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int pad = static_cast<int>(pad_size);

    int outH = H + 2 * pad;
    int outW = W + 2 * pad;

    // Output tensor shape [C, outH, outW]
    auto output = torch::empty({C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        C, H, W, pad,
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("median_dim0_forward", &median_dim0_forward,
          "Median along dim=0 for float32 tensor [N, C, H, W] -> [C, H, W]",
          pybind11::arg("input"));
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2D for float32 tensor [C, H, W] -> [C, H+2*pad, W+2*pad]",
          pybind11::arg("input"),
          pybind11::arg("pad_size"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);