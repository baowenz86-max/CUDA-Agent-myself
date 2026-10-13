#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    int64_t in_N_stride, int64_t in_C_stride,
    int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream);

extern "C" void logical_and_positive_less_than_one_launcher(
    const float* input,
    bool* output,
    int64_t total,
    cudaStream_t stream);

// 包装函数：双线性上采样
torch::Tensor upsample_bilinear2d(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    // 确保输入连续，简化 stride 处理（launcher 已支持任意 stride，但为了性能，我们传递原始 stride）
    // 如果输入非连续，launcher 会根据 stride 正确读取；此处不解包为连续，保留原始 stride。
    // 但为了安全，我们可以先连续化，避免复杂 stride 处理
    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H_in = x_contig.size(2);
    int W_in = x_contig.size(3);

    // 计算输出尺寸：PyTorch 的 nn.Upsample 使用 floor(scale_factor * size)
    int H_out = static_cast<int>(std::floor(H_in * scale_factor));
    int W_out = static_cast<int>(std::floor(W_in * scale_factor));

    // 分配输出张量（连续）
    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    upsample_bilinear2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        static_cast<float>(scale_factor),
        x_contig.stride(0), x_contig.stride(1),
        x_contig.stride(2), x_contig.stride(3),
        stream
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in upsample_bilinear2d: ", cudaGetErrorString(err));

    return output;
}

// 包装函数：logical_and (x > 0) & (x < 1)
torch::Tensor logical_and_positive_less_than_one(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // 确保输入连续
    auto x_contig = x.contiguous();
    int64_t total = x_contig.numel();

    // 分配输出 bool 张量（连续）
    auto output = torch::empty(x_contig.sizes(), x_contig.options().dtype(torch::kBool));

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    logical_and_positive_less_than_one_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        total,
        stream
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in logical_and_positive_less_than_one: ", cudaGetErrorString(err));

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Bilinear upsample with scale_factor (align_corners=False)",
          pybind11::arg("x"), pybind11::arg("scale_factor"));
    m.def("logical_and_positive_less_than_one", &logical_and_positive_less_than_one,
          "Logical AND of (x > 0) and (x < 1)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);