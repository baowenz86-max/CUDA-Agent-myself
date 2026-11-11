#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void sum_dim1_launcher(const float* input, float* output, int64_t B, int64_t C, int64_t H, int64_t W, cudaStream_t stream);

// ReLU forward binding
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), numel, stream);

    return output;
}

// Sum along dim=1 binding
// Input x shape [B, C, H, W], output shape [B, H, W]
// dim must be 1, as per contract
torch::Tensor sum_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "sum_dim_forward expects 4D tensor [B, C, H, W]");
    TORCH_CHECK(dim == 1, "sum_dim_forward only supports dim=1");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({B, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sum_dim1_launcher(x.data_ptr<float>(), output.data_ptr<float>(), B, C, H, W, stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
    m.def("sum_dim_forward", &sum_dim_forward, "Sum along dim=1",
          py::arg("x"), py::arg("dim") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
