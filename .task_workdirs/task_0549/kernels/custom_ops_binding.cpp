#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_clamp_max_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          float max_val,
                                          cudaStream_t stream);

extern "C" void custom_prelu_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       int C,
                                       int64_t HW,
                                       const float* weight,
                                       cudaStream_t stream);

// custom_clamp_max: elementwise max with scalar
torch::Tensor custom_clamp_max(torch::Tensor x, double max_val) {
    TORCH_CHECK(x.is_cuda(), "custom_clamp_max: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_clamp_max: x must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_clamp_max_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        static_cast<float>(max_val),
        stream
    );

    return output;
}

// custom_prelu: PReLU activation with channel-wise weight
torch::Tensor custom_prelu(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "custom_prelu: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "custom_prelu: weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_prelu: x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "custom_prelu: weight must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_prelu: x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "custom_prelu: weight must be 1D of size C");
    TORCH_CHECK(weight.is_contiguous(), "custom_prelu: weight must be contiguous");

    auto output = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t HW = x.size(2) * x.size(3);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_prelu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        static_cast<int>(C),
        HW,
        weight.data_ptr<float>(),
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_clamp_max", &custom_clamp_max,
          "Elementwise max with scalar",
          py::arg("x"), py::arg("max_val"));
    m.def("custom_prelu", &custom_prelu,
          "PReLU activation",
          py::arg("x"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
