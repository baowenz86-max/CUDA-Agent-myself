#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 外部 launcher 声明，与 custom_ops.cu 保持一致
extern "C" void sum_channel_dim_launcher(
    const float* x,
    float* out,
    int64_t batch,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

extern "C" void clip_01_launcher(
    const float* x,
    float* out,
    int64_t batch,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

// sum_channel_dim: 计算 dim=1 的和，keepdim=True
torch::Tensor sum_channel_dim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_channel_dim: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_channel_dim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_channel_dim: input must be 4D [batch, C, H, W]");

    // 确保输入连续，以便使用连续 strides
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t batch = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out = torch::empty({batch, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_channel_dim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        C,
        H,
        W,
        stream
    );

    return out;
}

// clip_01: 将输入裁剪到 [0,1]，支持任意 strides
torch::Tensor clip_01(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "clip_01: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clip_01: input must be float32");
    TORCH_CHECK(x.dim() == 4, "clip_01: input must be 4D [batch, 1, H, W]");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t batch = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clip_01_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        H,
        W,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sum_channel_dim", &sum_channel_dim, "Sum over channel dimension (dim=1, keepdim=True)",
          py::arg("x"));
    m.def("clip_01", &clip_01, "Clip values to [0, 1]",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
