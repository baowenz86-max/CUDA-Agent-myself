#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream);

extern "C" void square_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

static inline void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// Binding for reflection pad operator
// Contract: pad_forward(x: Tensor, pad_left:int, pad_right:int, pad_top:int, pad_bottom:int) -> Tensor
torch::Tensor pad_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    TORCH_CHECK(x.is_cuda(), "pad_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "pad_forward: x must be a 4D NCHW tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "pad_forward: x must be float32");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "pad_forward: padding values must be non-negative");
    TORCH_CHECK(pad_left < W && pad_right < W && pad_top < H && pad_bottom < H,
                "pad_forward: padding must be smaller than corresponding dimension");

    const int64_t outH = H + pad_top + pad_bottom;
    const int64_t outW = W + pad_left + pad_right;

    // Allocate contiguous output with same dtype/device as input
    auto options = torch::TensorOptions()
        .dtype(x.scalar_type())
        .device(x.device())
        .layout(torch::kStrided)
        .requires_grad(false);
    auto output = torch::empty({N, C, outH, outW}, options);

    // Input can be non-contiguous: use strides in elements
    // Note: tensors are row-major, but kernel expects strides in elements (as in .cu).
    const int64_t in_stride_n = x.stride(0);
    const int64_t in_stride_c = x.stride(1);
    const int64_t in_stride_h = x.stride(2);
    const int64_t in_stride_w = x.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    pad_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        stream
    );

    check_cuda_error("pad_forward");
    return output;
}

// Binding for element-wise square
// Contract: square_forward(x: Tensor) -> Tensor (same shape, dtype, device)
torch::Tensor square_forward(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "square_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "square_forward: x must be float32");

    // Output must have same shape, dtype, device; torch::empty_like preserves layout and strides.
    auto output = torch::empty_like(x);

    const int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    square_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    check_cuda_error("square_forward");
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("pad_forward", &pad_forward,
          "ReflectionPad2d forward (Padding: left, right, top, bottom)",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));

    m.def("square_forward", &square_forward,
          "Element-wise square forward",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
