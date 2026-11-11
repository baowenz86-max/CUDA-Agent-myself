#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void maxpool1d_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t L_in,
    int64_t L_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor maxpool1d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool ceil_mode) {
    // 输入检查
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D (N, C, L)");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // 提取尺寸
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L_in = input.size(2);

    // 计算输出长度（ceil_mode 当前实现不考虑，但传参保留）
    // 与 PyTorch MaxPool1d 公式一致
    int64_t L_out;
    if (ceil_mode) {
        // ceil_mode = true 时输出长度 = ceil((L + 2*padding - kernel_size)/stride) + 1
        // 但当前 CUDA kernel 未实现 ceil_mode，仅接受参数，实际仍按照 floor 计算。
        // 为保持与 kernel 一致，我们按 floor 计算，但契约中 ceil_mode 被调用为 false。
        // 这里安全处理：如果 ceil_mode 为 true，仍使用 floor 逻辑，但可能会与参考不一致。
        // 由于实际调用 ceil_mode = false，我们仅需支持 false 情况。
        // 如果未来需要支持，需要更新 kernel。暂时使用 floor 计算。
        TORCH_CHECK(false, "ceil_mode=true is not supported yet");
    } else {
        L_out = (L_in + 2 * padding - kernel_size) / stride + 1;
        // 确保有效窗口至少为 1，且不超过输入长度
        TORCH_CHECK(L_out > 0, "Output length must be positive");
    }

    // 分配输出张量
    auto output = torch::empty({N, C, L_out}, input.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    maxpool1d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        L_in,
        L_out,
        kernel_size,
        stride,
        padding,
        stream);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// 注册到 pybind
void register_maxpool1d(pybind11::module& m) {
    m.def("maxpool1d_forward", &maxpool1d_forward,
          "MaxPool1d forward",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("ceil_mode"));
}

// 自动注册
REGISTER_BINDING(maxpool1d, register_maxpool1d);
