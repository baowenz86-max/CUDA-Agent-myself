#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void custom_clamp_min_max_launcher(float* out, const float* in,
                                              long long n, cudaStream_t stream);
extern "C" void custom_triu_launcher(float* out, const float* in,
                                     int batch, int rows, int cols,
                                     cudaStream_t stream);
extern "C" void custom_add_bias_launcher(float* out, const float* in, const float* bias,
                                         int batch, int rows, int cols,
                                         cudaStream_t stream);

torch::Tensor custom_clamp_min_max(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_clamp_min_max_launcher(out.data_ptr<float>(), x.data_ptr<float>(),
                                  x.numel(), stream);
    return out;
}

torch::Tensor custom_triu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, rows, cols)");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_triu_launcher(out.data_ptr<float>(), x.data_ptr<float>(),
                         batch, rows, cols, stream);
    return out;
}

torch::Tensor custom_add_bias(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "Input x must be 3D (batch, rows, cols)");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D (rows, cols)");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    TORCH_CHECK(bias.size(0) == rows && bias.size(1) == cols,
                "bias shape mismatch with x's rows and cols");

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_add_bias_launcher(out.data_ptr<float>(), x.data_ptr<float>(), bias.data_ptr<float>(),
                             batch, rows, cols, stream);
    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_clamp_min_max", &custom_clamp_min_max, "Clamp each element to [0,1]",
          py::arg("x"));
    m.def("custom_triu", &custom_triu, "Upper triangular mask for each batch",
          py::arg("x"));
    m.def("custom_add_bias", &custom_add_bias, "Add bias broadcast over batch",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
