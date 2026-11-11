#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void mean_dim0_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    cudaStream_t stream
);

// mean_dim0 绑定
torch::Tensor mean_dim0(torch::Tensor x) {
    // 检查输入为 CUDA float32 张量
    TORCH_CHECK(x.is_cuda(), "mean_dim0: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim0: input must be float32");
    TORCH_CHECK(x.dim() >= 2, "mean_dim0: input must have at least 2 dimensions");
    TORCH_CHECK(x.size(-1) == 2, "mean_dim0: last dimension must be 2");

    int64_t ndim = x.dim();
    int64_t N = x.size(0);

    // 准备主机端 shape 和 strides 数组
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        strides[i] = x.stride(i);
    }

    // 分配输出张量：形状为去掉第一维后的 shape，连续内存
    std::vector<int64_t> out_shape(shape.begin() + 1, shape.end());
    auto out = torch::empty(out_shape, x.options());

    // 确保输出是连续的（torch::empty 默认连续）
    TORCH_CHECK(out.is_contiguous(), "mean_dim0: output is not contiguous");

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    mean_dim0_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        ndim,
        shape.data(),
        strides.data(),
        stream
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim0: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// view_as_complex_lastdim 绑定
torch::Tensor view_as_complex_lastdim(torch::Tensor x) {
    // 检查输入为 CUDA float32 张量
    TORCH_CHECK(x.is_cuda(), "view_as_complex_lastdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "view_as_complex_lastdim: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "view_as_complex_lastdim: input must have at least 1 dimension");
    TORCH_CHECK(x.size(-1) == 2, "view_as_complex_lastdim: last dimension must be 2");

    // 使用 torch::view_as_complex 构造复杂视图（复用存储）
    auto out = torch::view_as_complex(x);
    return out;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim0", &mean_dim0, "Mean over dimension 0", py::arg("x"));
    m.def("view_as_complex_lastdim", &view_as_complex_lastdim, "View input as complex tensor by pairing last dim", py::arg("x"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
