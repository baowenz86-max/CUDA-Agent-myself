#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void sum_dim1_launcher(const float* x,
                                  float* out,
                                  int N, int C, int H, int W,
                                  cudaStream_t stream);

extern "C" void relu_launcher(const float* x,
                               float* out,
                               int numel,
                               cudaStream_t stream);

// sum_dim1_forward: 对输入 [N, C, H, W] 沿 dim=1 求和，输出 [N, H, W]
torch::Tensor sum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C, H, W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "All dimensions must be positive");

    auto out = torch::empty({N, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sum_dim1_launcher(x.data_ptr<float>(),
                      out.data_ptr<float>(),
                      static_cast<int>(N),
                      static_cast<int>(C),
                      static_cast<int>(H),
                      static_cast<int>(W),
                      stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sum_dim1_launcher failed");

    return out;
}

// relu_forward: 对输入逐元素应用 ReLU
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_launcher(x.data_ptr<float>(),
                  out.data_ptr<float>(),
                  static_cast<int>(numel),
                  stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_launcher failed");

    return out;
}

// 注册绑定
void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1_forward", &sum_dim1_forward, "Sum over dimension 1",
          py::arg("x"));
    m.def("relu_forward", &relu_forward, "ReLU activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
