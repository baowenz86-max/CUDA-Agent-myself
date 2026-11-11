#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 文件中的 launcher
extern "C" void leaky_then_relu_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         float negative_slope,
                                         cudaStream_t stream);

// PyTorch 包装函数
torch::Tensor leaky_then_relu(torch::Tensor x, double negative_slope = 0.01) {
    // 检查输入
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor is empty");

    // 处理 non-contiguous 输入：显式连续化
    auto x_contig = x.contiguous();

    // 分配输出，匹配输入形状和 dtype
    auto output = torch::empty_like(x_contig);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    leaky_then_relu_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        static_cast<float>(negative_slope),
        stream
    );

    // 检查 kernel 启动错误（cudaGetLastError）
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("leaky_then_relu", &leaky_then_relu,
          "Fused leaky_relu followed by relu",
          pybind11::arg("x"),
          pybind11::arg("negative_slope") = 0.01);
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
