#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    const long long* input_strides,
    cudaStream_t stream
);

// PyTorch binding wrapper
torch::Tensor reflection_pad2d_forward(
    torch::Tensor input,
    int64_t padding_top,
    int64_t padding_bottom,
    int64_t padding_left,
    int64_t padding_right
) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    // Get dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Validate padding non-negative and within bounds to avoid undefined behavior
    TORCH_CHECK(padding_top >= 0 && padding_bottom >= 0 && padding_left >= 0 && padding_right >= 0,
                "Padding must be non-negative");
    // The kernel reflects modulo period, so padding can be arbitrary non-negative, but for safety we limit to reasonable size

    // Compute output dimensions
    int H_out = H_in + padding_top + padding_bottom;
    int W_out = W_in + padding_left + padding_right;

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // Get input strides (in elements)
    long long input_strides[4];
    input_strides[0] = input.stride(0);
    input_strides[1] = input.stride(1);
    input_strides[2] = input.stride(2);
    input_strides[3] = input.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        static_cast<int>(padding_top),
        static_cast<int>(padding_bottom),
        static_cast<int>(padding_left),
        static_cast<int>(padding_right),
        input_strides,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_reflection_pad2d(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection padding forward (CUDA)",
          py::arg("input"),
          py::arg("padding_top"),
          py::arg("padding_bottom"),
          py::arg("padding_left"),
          py::arg("padding_right"));
}

REGISTER_BINDING(reflection_pad2d, register_reflection_pad2d);
