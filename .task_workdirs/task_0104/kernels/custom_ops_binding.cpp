#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void leaky_relu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    float negative_slope,
    cudaStream_t stream);

torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor [N, C, H, W]");

    auto output = torch::empty_like(x);
    if (x.numel() == 0) {
        return output;
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t stride0 = x.stride(0);
    int64_t stride1 = x.stride(1);
    int64_t stride2 = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        N,
        C,
        H,
        stride0,
        stride1,
        stride2,
        static_cast<float>(negative_slope),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("x"),
          py::arg("negative_slope") = 0.01);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
