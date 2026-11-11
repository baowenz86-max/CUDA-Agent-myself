#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream);

extern "C" void cos_elementwise_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream);

// Wrapper for mean_dim1_keepdim
torch::Tensor mean_dim1_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mean_dim1_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim1_keepdim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_dim1_keepdim: input must be 4D (B, C, H, W)");
    x = x.contiguous();

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C > 0, "mean_dim1_keepdim: channel dimension must be positive");

    auto out = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim1_keepdim kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Wrapper for cos_elementwise
torch::Tensor cos_elementwise(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "cos_elementwise: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "cos_elementwise: input must be float32");
    x = x.contiguous();

    int64_t numel = x.numel();
    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cos_elementwise_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cos_elementwise kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops_binding(pybind11::module& m) {
    m.def("mean_dim1_keepdim", &mean_dim1_keepdim, "Mean over channel dimension with keepdim=True",
          pybind11::arg("x"));
    m.def("cos_elementwise", &cos_elementwise, "Elementwise cosine",
          pybind11::arg("x"));
}

// Auto-register this binding
REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
