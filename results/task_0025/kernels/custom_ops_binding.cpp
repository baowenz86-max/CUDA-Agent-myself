#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad,
    cudaStream_t stream);

extern "C" void erf_launcher(const float* input, float* output, size_t n, cudaStream_t stream);
extern "C" void sin_launcher(const float* input, float* output, size_t n, cudaStream_t stream);

// 反射填充包装函数
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t pad) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(pad >= 0, "pad must be non-negative");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    TORCH_CHECK(pad < H_in && pad < W_in, "pad must be smaller than input spatial dims");

    int64_t H_out = H_in + 2 * pad;
    int64_t W_out = W_in + 2 * pad;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H_in),
        static_cast<int>(W_in),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        static_cast<int>(pad),
        stream);

    return output;
}

// erf 包装函数
torch::Tensor erf_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    erf_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<size_t>(input.numel()),
        stream);

    return output;
}

// sin 包装函数
torch::Tensor sin_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sin_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<size_t>(input.numel()),
        stream);

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          py::arg("input"), py::arg("pad"));
    m.def("erf_forward", &erf_forward, "Erf forward", py::arg("input"));
    m.def("sin_forward", &sin_forward, "Sin forward", py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
