#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void tril_launcher(const float* input, float* output,
                              int N, int C, int H, int W,
                              cudaStream_t stream);

extern "C" void relu_launcher(const float* input, float* output,
                              int total_elements,
                              cudaStream_t stream);

// Wrapper for torch.tril on last two dimensions
torch::Tensor tril(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // Expected 4D tensor: [N, C, H, W]
    TORCH_CHECK(x.dim() == 4, "Input must be 4D, got ", x.dim());

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tril_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                  N, C, H, W, stream);

    return output;
}

// Wrapper for F.relu (element-wise max(0, x))
torch::Tensor relu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_launcher(x.data_ptr<float>(), output.data_ptr<float>(),
                  x.numel(), stream);

    return output;
}

// Register both operators with pybind11
void register_custom_ops(pybind11::module& m) {
    m.def("tril", &tril, "Apply torch.tril on last two dimensions",
          py::arg("x"));
    m.def("relu", &relu, "Apply ReLU activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
