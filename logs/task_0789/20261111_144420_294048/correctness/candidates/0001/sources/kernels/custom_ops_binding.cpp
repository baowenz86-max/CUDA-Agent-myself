#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void reduce_mean_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void scalar_mul_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    float scalar,
    cudaStream_t stream);

torch::Tensor reduce_mean_dim1_keepdim_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must have 4 dimensions (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Match reference output shape [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reduce_mean_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H,
        W,
        stream);

    // Check for kernel launch errors (optional, but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor scalar_mul_forward(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    scalar_mul_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        static_cast<float>(scalar),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("reduce_mean_dim1_keepdim", &reduce_mean_dim1_keepdim_forward,
          "Compute mean over dim=1 with keepdim=True",
          py::arg("x"));
    m.def("scalar_mul", &scalar_mul_forward,
          "Multiply input tensor by a scalar",
          py::arg("x"), py::arg("scalar"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
