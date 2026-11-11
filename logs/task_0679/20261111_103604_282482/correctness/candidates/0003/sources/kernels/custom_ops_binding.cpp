#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <climits>
#include <cfloat>
#include "../binding_registry.h"

extern "C" void torch_sum_dim1_launcher(const float* input, float* output, int64_t batch, int64_t height, int64_t width, cudaStream_t stream);
extern "C" void torch_argmax_all_launcher(const float* input, int64_t* output, int64_t numel, cudaStream_t stream);

torch::Tensor torch_sum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, C, H, W]");
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C == 3, "Channel dimension must be 3 per contract");
    auto output = torch::empty({B, H, W}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    torch_sum_dim1_launcher(x.data_ptr<float>(), output.data_ptr<float>(), B, H, W, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor torch_argmax_all_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty({}, x.options().dtype(torch::kInt64));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    torch_argmax_all_launcher(x.data_ptr<float>(), output.data_ptr<int64_t>(), x.numel(), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("torch_sum_dim1", &torch_sum_dim1_forward, "Sum over dimension 1", pybind11::arg("x"));
    m.def("torch_argmax_all", &torch_argmax_all_forward, "Argmax over all elements", pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
