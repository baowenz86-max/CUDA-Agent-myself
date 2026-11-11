#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

// MaxPool2d forward binding
torch::Tensor maxpool2d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "Input height and width must be even");

    auto output = torch::empty({N, C, H / 2, W / 2}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

// Softmax over dim=1 binding
torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "MaxPool2d forward",
          py::arg("x"));
    m.def("softmax_dim1_forward", &softmax_dim1_forward,
          "Softmax over channel dimension",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
