#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// 前向声明来自 custom_ops.cu 的 launcher（extern "C"）
extern "C" void channel_sum_launcher(
    const float* x,
    float* y,
    int B,
    int C,
    int H,
    int W,
    long long sB,
    long long sC,
    long long sH,
    long long sW,
    long long out_sB,
    long long out_sH,
    long long out_sW,
    cudaStream_t stream
);

extern "C" void exp_launcher(
    const float* x,
    float* y,
    long long total,
    cudaStream_t stream
);

// channel_sum: 沿 dim=1 求和，输出形状 [B, H, W]
torch::Tensor channel_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "channel_sum: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "channel_sum: input must be float32");
    TORCH_CHECK(x.dim() == 4, "channel_sum: input must be 4D");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // 输出形状 [B, H, W]
    auto y = torch::empty({B, H, W}, x.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 使用输入的 strides（以元素为单位，与 launcher 的 long long 类型匹配）
    channel_sum_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, C, H, W,
        (long long)x.stride(0),
        (long long)x.stride(1),
        (long long)x.stride(2),
        (long long)x.stride(3),
        (long long)y.stride(0),
        (long long)y.stride(1),
        (long long)y.stride(2),
        stream
    );

    // 检查 kernel launch 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "channel_sum kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// exp_forward: 逐元素指数（避免与全局 ::exp 冲突）
torch::Tensor exp_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "exp: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "exp: input must be float32");

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    exp_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (long long)x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "exp kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("channel_sum", &channel_sum, "Sum over channel dimension (dim=1)",
          py::arg("x"));
    m.def("exp", &exp_forward, "Element-wise exponential",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
