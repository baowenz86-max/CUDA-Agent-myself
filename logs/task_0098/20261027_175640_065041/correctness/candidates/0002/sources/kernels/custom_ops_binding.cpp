#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void reflection_pad2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad,
    long long sn, long long sc, long long sh, long long sw,
    cudaStream_t stream);

extern "C" void elementwise_max_launcher(
    const float* a, const float* b, float* out,
    int total,
    cudaStream_t stream);

extern "C" void elementwise_minimum_launcher(
    const float* a, const float* b, float* out,
    int total,
    cudaStream_t stream);

// 包装函数：reflection_pad2d
torch::Tensor reflection_pad2d(torch::Tensor input, int64_t pad_size) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(pad_size >= 0, "pad_size must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int H_out = H_in + 2 * pad_size;
    int W_out = W_in + 2 * pad_size;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out, static_cast<int>(pad_size),
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        stream);

    return output;
}

// 包装函数：elementwise_max
torch::Tensor elementwise_max(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto output = torch::empty_like(a);
    int total = a.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elementwise_max_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

// 包装函数：elementwise_minimum
torch::Tensor elementwise_minimum(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto output = torch::empty_like(a);
    int total = a.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elementwise_minimum_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d", &reflection_pad2d,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("pad_size"));
    m.def("elementwise_max", &elementwise_max,
          "Elementwise maximum",
          py::arg("a"), py::arg("b"));
    m.def("elementwise_minimum", &elementwise_minimum,
          "Elementwise minimum",
          py::arg("a"), py::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
