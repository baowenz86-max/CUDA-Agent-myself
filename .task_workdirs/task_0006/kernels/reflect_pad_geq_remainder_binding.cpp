#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void reflect_pad_geq_remainder_launcher(
    const float* input,
    long long* output,
    int N, int C, int H, int W,
    int H_new, int W_new,
    int top_pad, int left_pad,
    cudaStream_t stream
);

// PyTorch wrapper
torch::Tensor reflect_pad_geq_remainder_forward(
    torch::Tensor input,
    int left_pad,
    int right_pad,
    int top_pad,
    int bottom_pad
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(left_pad >= 0 && right_pad >= 0 && top_pad >= 0 && bottom_pad >= 0,
                "Padding values must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Check that padding is valid for reflection: pad size must be less than dim size
    TORCH_CHECK(left_pad < W && right_pad < W, "Padding must be less than input width");
    TORCH_CHECK(top_pad < H && bottom_pad < H, "Padding must be less than input height");

    int H_new = H + top_pad + bottom_pad;
    int W_new = W + left_pad + right_pad;

    // Allocate output tensor with int64 dtype, matching torch.ge(...).remainder(2) result.
    auto options = torch::TensorOptions().dtype(torch::kLong).device(input.device());
    auto output = torch::empty({N, C, H_new, W_new}, options);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher with explicit long long cast to match launcher signature.
    reflect_pad_geq_remainder_launcher(
        input.data_ptr<float>(),
        static_cast<long long*>(output.data_ptr<int64_t>()),
        N, C, H, W,
        H_new, W_new,
        top_pad, left_pad,
        stream
    );

    return output;
}

// Registration function
void register_reflect_pad_geq_remainder(pybind11::module& m) {
    m.def("reflect_pad_geq_remainder_forward", &reflect_pad_geq_remainder_forward,
          "Reflection pad, then ge>=0 and remainder(2)",
          py::arg("input"),
          py::arg("left_pad"),
          py::arg("right_pad"),
          py::arg("top_pad"),
          py::arg("bottom_pad"));
}

// Auto-register
REGISTER_BINDING(reflect_pad_geq_remainder, register_reflect_pad_geq_remainder);