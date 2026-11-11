#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_dim1_launcher(const float* input,
                                           float* output,
                                           int B, int C, int H, int W,
                                           cudaStream_t stream);

extern "C" void custom_gelu_launcher(const float* input,
                                      float* output,
                                      int N,
                                      cudaStream_t stream);

static torch::Tensor custom_mean_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_mean_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean_dim1: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_mean_dim1: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "custom_mean_dim1: input must be 4D [B,C,H,W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    return output;
}

static torch::Tensor custom_gelu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_gelu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_gelu: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_gelu: input must be contiguous");

    int64_t N = x.numel();
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gelu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        stream
    );

    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean_dim1", &custom_mean_dim1, "Mean over channel dim (keepdim=True)",
          py::arg("x"));
    m.def("custom_gelu", &custom_gelu, "GELU activation (tanh approximation)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
