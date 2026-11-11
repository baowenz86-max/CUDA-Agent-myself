#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 中定义的 launcher
extern "C" void mean_reduce_op_launcher(const float* input,
                                        float* output,
                                        int64_t batch,
                                        int64_t inner_size,
                                        int64_t batch_stride,
                                        cudaStream_t stream);

extern "C" void remainder_op_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      double divisor,
                                      cudaStream_t stream);

// mean_reduce_op 包装函数
torch::Tensor mean_reduce_op(torch::Tensor x, int64_t dim, bool keepdim) {
    // 检查输入
    TORCH_CHECK(x.is_cuda(), "mean_reduce_op: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_reduce_op: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_reduce_op: input must be 4-dimensional");
    // 检查 dim 是否为 1（对应通道维），keepdim 必须为 true（本任务固定）
    TORCH_CHECK(dim == 1, "mean_reduce_op: only dim=1 reduction is supported");
    TORCH_CHECK(keepdim, "mean_reduce_op: keepdim must be true");

    // 确定参数
    auto batch = x.size(0);
    auto inner_size = x.size(1) * x.size(2) * x.size(3);
    // 输入可能非连续，使用 strides[0] 作为 batch 维的步长（元素个数）
    auto batch_stride = x.stride(0);

    // 分配输出，形状 [batch, 1, 1, 1]
    auto output = torch::empty({batch, 1, 1, 1}, x.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    mean_reduce_op_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        inner_size,
        batch_stride,
        stream
    );

    // 检查 kernel 启动错误（可选但推荐）
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_reduce_op: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// remainder_op 包装函数
torch::Tensor remainder_op(torch::Tensor x, double divisor) {
    // 检查输入
    TORCH_CHECK(x.is_cuda(), "remainder_op: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "remainder_op: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "remainder_op: input must be contiguous");

    auto numel = x.numel();
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    remainder_op_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        divisor,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "remainder_op: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("mean_reduce_op", &mean_reduce_op,
          "Mean reduce over channel and spatial dimensions",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("remainder_op", &remainder_op,
          "Element-wise remainder",
          py::arg("x"), py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
