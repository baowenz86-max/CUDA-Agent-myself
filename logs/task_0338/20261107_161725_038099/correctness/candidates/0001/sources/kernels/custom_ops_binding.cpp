#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 中的 launcher
extern "C" void custom_ge_forward_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    float threshold,
    cudaStream_t stream
);

// PyTorch 包装函数
static torch::Tensor custom_ge_forward(torch::Tensor x, double threshold) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x, x.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_ge_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        x.numel(),
        static_cast<float>(threshold),
        stream
    );

    return output;
}

void register_custom_ge(pybind11::module& m) {
    m.def("custom_ge_forward", &custom_ge_forward,
          "Custom elementwise greater-equal forward (CUDA)",
          py::arg("x"), py::arg("threshold"));
}

REGISTER_BINDING(custom_ge, register_custom_ge);
