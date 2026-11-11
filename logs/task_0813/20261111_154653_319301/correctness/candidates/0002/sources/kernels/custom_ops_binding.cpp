#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void ceil_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream);

extern "C" void reflection_pad2d_forward_launcher(const float* input,
                                                   float* output,
                                                   int N, int C, int H, int W,
                                                   int pad_left, int pad_right, int pad_top, int pad_bottom,
                                                   cudaStream_t stream);

// Wrapper for ceil_forward
torch::Tensor ceil_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "ceil_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "ceil_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "ceil_forward: input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    ceil_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "ceil_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for reflection_pad2d_forward
torch::Tensor reflection_pad2d_forward(torch::Tensor x,
                                       int64_t pad_left,
                                       int64_t pad_right,
                                       int64_t pad_top,
                                       int64_t pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "reflection_pad2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reflection_pad2d_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "reflection_pad2d_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "reflection_pad2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "reflection_pad2d_forward: padding sizes must be non-negative");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int out_h = H + pad_top + pad_bottom;
    int out_w = W + pad_left + pad_right;

    auto output = torch::empty({N, C, out_h, out_w}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reflection_pad2d_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("ceil_forward", &ceil_forward,
          "Element-wise ceil for float32 CUDA tensors",
          py::arg("x"));
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection padding for 2D spatial dimensions",
          py::arg("x"),
          py::arg("pad_left") = 3,
          py::arg("pad_right") = 3,
          py::arg("pad_top") = 3,
          py::arg("pad_bottom") = 3);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
