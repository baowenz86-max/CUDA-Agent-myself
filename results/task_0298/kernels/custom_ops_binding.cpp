#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

// PyTorch 包装函数
torch::Tensor gelu_forward(torch::Tensor x) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int64_t numel = x.numel();
    auto output = torch::empty_like(x);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 CUDA launcher
    gelu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward (approximate)",
          py::arg("x"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
