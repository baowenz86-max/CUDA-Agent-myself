#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// 前向声明来自 custom_ops.cu 的 launcher
// 该 launcher 现在执行完整的卷积+ReLU，并处理 bias
extern "C" void conv2d_relu_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t R, int64_t S,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups,
    cudaStream_t stream);

// 辅助：检查CUDA错误
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor conv2d_relu(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    // 输入检查
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels/groups, kh, kw)");
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
    }

    // 确保张量连续
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    c10::optional<torch::Tensor> bias_contig;
    if (bias.has_value() && bias->defined()) {
        bias_contig = bias->contiguous();
    }

    // 获取尺寸
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);
    int64_t K = weight_contig.size(0); // out_channels
    int64_t C_per_group = weight_contig.size(1);
    int64_t R = weight_contig.size(2);
    int64_t S = weight_contig.size(3);

    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(C == groups * C_per_group, "input channels must equal groups * weight in_channels");
    if (bias_contig.has_value() && bias_contig->defined()) {
        TORCH_CHECK(bias_contig->numel() == K, "bias size must equal out_channels");
    }

    // 输出尺寸计算
    int64_t out_h = (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding - dilation * (S - 1) - 1) / stride + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "invalid output size");

    auto output = torch::empty({N, K, out_h, out_w}, input_contig.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher 执行卷积+ReLU
    const float* bias_ptr = nullptr;
    if (bias_contig.has_value() && bias_contig->defined()) {
        bias_ptr = bias_contig->data_ptr<float>();
    }

    conv2d_relu_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        N, C, H, W,
        K, R, S,
        stride, padding, dilation,
        groups,
        stream
    );

    // 检查 kernel 启动错误
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(std::string("conv2d_relu_launcher failed: ") + cudaGetErrorString(launch_err));
    }

    return output;
}

void register_conv2d_relu(pybind11::module& m) {
    m.def("conv2d_relu", &conv2d_relu,
          "Conv2d + ReLU (custom CUDA kernel)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}

REGISTER_BINDING(conv2d_relu, register_conv2d_relu);