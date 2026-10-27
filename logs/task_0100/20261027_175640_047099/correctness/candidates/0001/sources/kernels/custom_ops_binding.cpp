#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 文件中的 launcher（使用原始指针和 cudaStream_t）
extern "C" void upsample_bilinear2d_launcher(const float* input,
                                             float* output,
                                             int N, int C, int H, int W,
                                             cudaStream_t stream);

extern "C" void squeeze_copy_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

// -----------------------------------------------------------------------------
// PyTorch 包装器：upsample_bilinear2d
// 参数：x：CUDA float32 4D tensor [N, C, H, W]
// 返回：新 tensor [N, C, H*2, W*2]
// -----------------------------------------------------------------------------
torch::Tensor upsample_bilinear2d(torch::Tensor x) {
    // 检查设备
    TORCH_CHECK(x.is_cuda(), "upsample_bilinear2d: input must be a CUDA tensor");
    // 检查 dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "upsample_bilinear2d: input must be float32");
    // 检查维度
    TORCH_CHECK(x.dim() == 4, "upsample_bilinear2d: input must be 4D (N, C, H, W)");

    // 确保输入连续（如果支持 strides 也可以继续，但 cuDNN 需要连续）
    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    // 分配输出 tensor
    auto output = torch::empty({N, C, H * 2, W * 2}, x_contig.options());

    // 获取当前 CUDA 流
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    upsample_bilinear2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// PyTorch 包装器：squeeze_copy
// 参数：x：CUDA float32 4D tensor [N, C, H, W]
// 返回：与输入相同 shape 的新 tensor（深拷贝）
// -----------------------------------------------------------------------------
torch::Tensor squeeze_copy(torch::Tensor x) {
    // 检查设备
    TORCH_CHECK(x.is_cuda(), "squeeze_copy: input must be a CUDA tensor");
    // 检查 dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "squeeze_copy: input must be float32");
    // 检查维度（本任务固定为 4D）
    TORCH_CHECK(x.dim() == 4, "squeeze_copy: input must be 4D (N, C, H, W)");

    // 确保输入连续（副本使用连续内存复制）
    auto x_contig = x.contiguous();

    int64_t numel = x_contig.numel();

    // 分配输出 (同 shape)
    auto output = torch::empty_like(x_contig);

    // 获取当前 CUDA 流
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    squeeze_copy_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// 注册函数
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          "Upsample bilinear 2x (NCHW, float32)",
          pybind11::arg("x"));
    m.def("squeeze_copy", &squeeze_copy,
          "Squeeze copy (NCHW, float32)",
          pybind11::arg("x"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
