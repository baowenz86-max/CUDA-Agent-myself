#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher 函数
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void custom_max_pool2d_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation,
    cudaStream_t stream);

// 自定义绝对值算子
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher，handle 错误
    custom_abs_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    // 检查 launch 错误（cudaPeekAtLastError）
    cudaError_t err = cudaPeekAtLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

// 自定义最大池化算子
torch::Tensor custom_max_pool2d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // 根据 PyTorch 的 MaxPool2d 计算输出尺寸（floor 模式）
    int64_t out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, out_h, out_w}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_pool2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        C,
        H,
        W,
        kernel_size,
        kernel_size,  // kernel_h, kernel_w 相同
        stride,
        padding,
        padding,     // pad_h, pad_w 相同（但契约中 padding 是两个元素的元组，这里暂时假设对称）
        dilation,
        stream
    );

    cudaError_t err = cudaPeekAtLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

// 注册到 pybind
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Custom absolute value",
          py::arg("x"));

    m.def("custom_max_pool2d", &custom_max_pool2d, "Custom max pooling 2D",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("ceil_mode"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
