#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

extern "C" void cumsum_dim1_forward_launcher(const float* input,
                                              float* output,
                                              int64_t B, int64_t C, int64_t H, int64_t W,
                                              cudaStream_t stream);

// Wrapper for gelu_forward
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gelu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Wrapper for cumsum_dim1_forward
torch::Tensor cumsum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "cumsum_dim1_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "cumsum_dim1_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "cumsum_dim1_forward: input must be 4D (B, C, H, W)");

    auto output = torch::empty_like(x);
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cumsum_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward,
          "GELU activation forward (tanh approximation)",
          py::arg("x"));
    m.def("cumsum_dim1_forward", &cumsum_dim1_forward,
          "Cumulative sum along dimension 1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);