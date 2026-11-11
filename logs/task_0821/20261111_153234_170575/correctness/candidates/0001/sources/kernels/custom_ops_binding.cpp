#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream);

extern "C" void argsort_dim1_launcher(
    const float* input,
    long long* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Reflection padding forward
static torch::Tensor reflection_pad2d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0,
                "Padding values must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(pad_left < W && pad_right < W && pad_top < H && pad_bottom < H,
                "Padding must be less than input dimensions for reflection");

    int64_t H_out = H + pad_top + pad_bottom;
    int64_t W_out = W + pad_left + pad_right;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(pad_left), static_cast<int>(pad_right),
        static_cast<int>(pad_top), static_cast<int>(pad_bottom),
        stream);

    return output;
}

// Argsort along dimension 1 (channel)
static torch::Tensor argsort_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for this task");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Output is int64 indices with same shape
    auto output = torch::empty({N, C, H, W},
                               x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    argsort_dim1_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<long long*>(output.data_ptr<int64_t>()),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W),
        stream);

    return output;
}

static void register_custom_ops_binding(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forwards (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("pad_left"),
          pybind11::arg("pad_right"),
          pybind11::arg("pad_top"),
          pybind11::arg("pad_bottom"));

    m.def("argsort_forward", &argsort_forward,
          "Argsort along dim=1 (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
