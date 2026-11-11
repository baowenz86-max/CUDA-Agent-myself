// Binding for clamp and maxpool custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void clamp_forward_launcher(float* out, const float* in, long long numel, cudaStream_t stream);
extern "C" void maxpool2d_forward_launcher(float* out, const float* in,
                                           int N, int C, int H, int W, cudaStream_t stream);

// clamp_forward binding
static torch::Tensor clamp_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "clamp_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clamp_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "clamp_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    long long numel = static_cast<long long>(x.numel());
    if (numel == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// maxpool2d_forward binding
static torch::Tensor maxpool2d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "maxpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "maxpool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "maxpool2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(), "maxpool2d_forward: input must be contiguous");

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "maxpool2d_forward: H and W must be even for 2x2/2 pool");

    int OH = H / 2;
    int OW = W / 2;
    auto output = torch::empty({N, C, OH, OW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    maxpool2d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool2d_forward kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Registration
static void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward, "Elementwise clamp to [0,1]",
          py::arg("x"));
    m.def("maxpool2d_forward", &maxpool2d_forward, "2x2 max pooling stride 2",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
