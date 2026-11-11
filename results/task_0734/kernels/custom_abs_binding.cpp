#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void custom_abs_launcher(const float* input, float* output,
                                    int64_t numel, cudaStream_t stream);

torch::Tensor custom_abs_forward(torch::Tensor x) {
    // 输入检查：必须是 CUDA 张量且 dtype 为 float32
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    // 处理任意 strides：如果输入不是 contiguous，则显式转换为 contiguous
    // 这样 kernel 可以安全地按线性顺序访问元素
    torch::Tensor x_contig = x.is_contiguous() ? x : x.contiguous();

    // 分配输出张量，与输入具有相同的 shape/dtype/device/options；
    // torch::empty_like 使用 x_contig 的 options，保证输出也是 contiguous 的
    auto output = torch::empty_like(x_contig);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    custom_abs_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    // 检查 kernel launch 后的错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

void register_custom_abs(pybind11::module& m) {
    m.def("custom_abs", &custom_abs_forward,
          "Element-wise absolute value (torch.abs) on CUDA",
          py::arg("x"));
}

REGISTER_BINDING(custom_abs, register_custom_abs);
