#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 custom_ops.cu 中的 launcher
extern "C" void copy_to_cuda_launcher(
    float* dst_cuda,
    const float* src_cpu,
    int64_t numel,
    cudaStream_t stream);

extern "C" void pad_launcher(
    const float* src,
    float* dst,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pad,
    cudaStream_t stream);

extern "C" void sum_dim_launcher(
    const float* src,
    float* dst,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t dim,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// copy_to_cuda_forward: 将 CPU float32 连续张量复制到当前 CUDA 设备
// 如果输入已经是 CUDA 张量，则直接返回（等价于 x.cuda() 对 CUDA 张量的行为）
// -----------------------------------------------------------------------------
torch::Tensor copy_to_cuda_forward(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "copy_to_cuda_forward: input dtype must be float32");
    TORCH_CHECK(x.is_contiguous(), "copy_to_cuda_forward: input must be contiguous");

    // 如果输入已经在 CUDA 上，直接返回
    if (x.is_cuda()) {
        return x;
    }

    int64_t numel = x.numel();
    auto output = torch::empty_like(x, x.options().device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    copy_to_cuda_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "copy_to_cuda_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// pad_forward: 对输入四维张量 [N, C, H, W] 在 H 和 W 维度各填充 pad 个零
// -----------------------------------------------------------------------------
torch::Tensor pad_forward(torch::Tensor x, int64_t pad) {
    TORCH_CHECK(x.is_cuda(), "pad_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "pad_forward: input dtype must be float32");
    TORCH_CHECK(x.is_contiguous(), "pad_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "pad_forward: input must be 4-dimensional");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t out_H = H + 2 * pad;
    int64_t out_W = W + 2 * pad;

    auto output = torch::zeros({N, C, out_H, out_W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    pad_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        pad,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "pad_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// sum_dim_forward: 沿指定维度 dim 求和
// -----------------------------------------------------------------------------
torch::Tensor sum_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "sum_dim_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim_forward: input dtype must be float32");
    TORCH_CHECK(x.is_contiguous(), "sum_dim_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "sum_dim_forward: input must be 4-dimensional");
    TORCH_CHECK(dim == 2, "sum_dim_forward: only dim=2 is supported");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::zeros({N, C, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        dim,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim_forward: CUDA error: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// 注册所有算子
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("copy_to_cuda_forward", &copy_to_cuda_forward,
          "Copy a CPU float32 tensor to current CUDA device",
          py::arg("x"));
    m.def("pad_forward", &pad_forward,
          "Pad a CUDA float32 tensor with zeros on H and W dimensions",
          py::arg("x"), py::arg("pad"));
    m.def("sum_dim_forward", &sum_dim_forward,
          "Sum over a specified dimension of a CUDA float32 tensor",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
