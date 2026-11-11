#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 中导出的 launcher
extern "C" void mean_dim_launcher(const float* x, float* out, int B, int C, int H, int W, cudaStream_t stream);
extern "C" void cosine_similarity_launcher(const float* x1, const float* x2, float* out, int B, int D1, int D2, cudaStream_t stream);

// 包装函数：沿指定维度计算均值，只支持 dim=1
torch::Tensor mean_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported for mean_dim_forward");
    TORCH_CHECK(x.dim() == 4, "Input tensor must have 4 dimensions [B, C, H, W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // 分配输出 [B, H, W]
    auto out = torch::empty({B, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// 包装函数：计算余弦相似度，沿 dim=1
torch::Tensor cosine_similarity_forward(torch::Tensor x1, torch::Tensor x2) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "Both input tensors must be CUDA tensors");
    TORCH_CHECK(x1.dtype() == torch::kFloat32 && x2.dtype() == torch::kFloat32, "Both input tensors must be float32");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous(), "Both input tensors must be contiguous");
    TORCH_CHECK(x1.dim() == 3 && x2.dim() == 3, "Input tensors must have 3 dimensions [B, D1, D2]");
    TORCH_CHECK(x1.size(0) == x2.size(0), "Batch dimensions must match");
    TORCH_CHECK(x1.size(1) == x2.size(1), "Feature dimensions (D1) must match");
    TORCH_CHECK(x1.size(2) == x2.size(2), "Spatial dimensions (D2) must match");

    int64_t B = x1.size(0);
    int64_t D1 = x1.size(1);
    int64_t D2 = x1.size(2);

    // 分配输出 [B, D2]
    auto out = torch::empty({B, D2}, x1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cosine_similarity_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(D1),
        static_cast<int>(D2),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cosine_similarity_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim_forward", &mean_dim_forward,
          "Compute mean along dimension 1 (C) for a 4D tensor",
          py::arg("x"), py::arg("dim"));
    m.def("cosine_similarity_forward", &cosine_similarity_forward,
          "Compute cosine similarity along dimension 1",
          py::arg("x1"), py::arg("x2"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
