#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 custom_ops.cu 中导出的 launcher
extern "C" void upsample_bilinear2x_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream
);

extern "C" void ceil_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream
);

// 包装函数：upsample_bilinear2x_forward
torch::Tensor upsample_bilinear2x_forward(torch::Tensor input) {
    // 输入检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");

    // 确保输入连续
    input = input.contiguous();

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int outH = H * 2;
    int outW = W * 2;

    // 创建连续输出张量
    auto output = torch::empty({N, C, outH, outW}, input.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher（不再传递 stride 指针）
    upsample_bilinear2x_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        stream
    );

    // 检查 kernel 启动错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "upsample_bilinear2x_launcher failed");

    return output;
}

// 包装函数：ceil_forward
torch::Tensor ceil_forward(torch::Tensor input) {
    // 输入检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    // 确保输入连续
    input = input.contiguous();

    // 创建连续输出张量，保留 shape
    auto output = torch::empty_like(input);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    ceil_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    // 检查 kernel 启动错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ceil_launcher failed");

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2x_forward", &upsample_bilinear2x_forward,
          "Upsample bilinear 2x forward",
          py::arg("input"));
    m.def("ceil_forward", &ceil_forward,
          "Ceil forward",
          py::arg("input"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
