#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launchers from custom_ops.cu
extern "C" void trunc_launcher(float* output,
                                const float* input,
                                int64_t numel,
                                cudaStream_t stream);

extern "C" void max_pool2d_launcher(const float* input,
                                     float* output,
                                     int64_t N,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     cudaStream_t stream);

// trunc_forward (torch.trunc)
torch::Tensor trunc_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must have at least one element");
    // Ensure contiguous for kernel indexing
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    trunc_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "trunc_launcher failed with error: ", cudaGetErrorString(err));
    return output;
}

// max_pool2d_forward (F.max_pool2d with kernel_size=2, stride=2)
torch::Tensor max_pool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(kernel_size == 2, "Only kernel_size=2 is supported");
    TORCH_CHECK(stride == 2, "Only stride=2 is supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "H and W must be even for kernel=2 stride=2 without padding");
    // Ensure contiguous for kernel indexing
    auto x_contig = x.contiguous();
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t Ho = H / 2;
    int64_t Wo = W / 2;
    auto output = torch::empty({N, C, Ho, Wo}, x_contig.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        H,
        W,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_pool2d_launcher failed with error: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("trunc_forward", &trunc_forward, "trunc_forward (CUDA)",
          pybind11::arg("x"));
    m.def("max_pool2d_forward", &max_pool2d_forward, "max_pool2d_forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("kernel_size"), pybind11::arg("stride"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
