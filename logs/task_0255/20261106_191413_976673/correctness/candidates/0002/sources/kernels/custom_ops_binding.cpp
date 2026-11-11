#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// 声明 .cu 中导出的 launcher
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t in_height,
    int64_t in_width,
    int64_t out_height,
    int64_t out_width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// avgpool2d_forward 包装
// -----------------------------------------------------------------------------
torch::Tensor avgpool2d_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "avgpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "avgpool2d_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "avgpool2d_forward: input must be 4D (N,C,H,W)");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    const int kernel_size = 3;
    const int stride = 3;
    const int padding = 0;
    int64_t out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
    int64_t out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

    // 根据契约，输出为连续 NCHW 张量
    auto output = torch::empty({batch, channels, out_height, out_width}, input.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 传递 strides（元素数）
    int64_t in_stride_batch = input.stride(0);
    int64_t in_stride_channel = input.stride(1);
    int64_t in_stride_height = input.stride(2);
    int64_t in_stride_width = input.stride(3);

    avgpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "avgpool2d_forward: kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// add_bias_forward 包装
// -----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "add_bias_forward: input must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "add_bias_forward: bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "add_bias_forward: input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "add_bias_forward: bias must be float32");
    TORCH_CHECK(input.dim() == 4, "add_bias_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3, "add_bias_forward: bias must be 3D (C,1,1)");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    TORCH_CHECK(channels == bias.size(0), "add_bias_forward: channel dimension mismatch");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "add_bias_forward: bias must have shape [C,1,1]");

    // 输出与输入具有相同的 shape 和 strides
    auto output = torch::empty_like(input);

    int64_t in_stride_batch = input.stride(0);
    int64_t in_stride_channel = input.stride(1);
    int64_t in_stride_height = input.stride(2);
    int64_t in_stride_width = input.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias_forward: kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// 注册绑定
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward, "Average pooling 2d forward (kernel_size=3, stride=3)");
    m.def("add_bias_forward", &add_bias_forward, "Add bias to input (broadcast along channels)");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
