#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void leaky_relu_launcher(
    float* out,
    const float* in,
    long long numel,
    int ndim,
    const long long* shape,
    const long long* in_strides,
    const long long* out_strides,
    float negative_slope,
    cudaStream_t stream);

extern "C" void max_pool2d_2x2_stride2_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* in_strides,
    const long long* out_strides,
    cudaStream_t stream);

extern "C" void conv2d_launcher(
    float* out,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int use_bias,
    cudaStream_t stream);

// Helper to convert torch::Tensor to vector of long long for shape/strides
static std::vector<long long> get_strides(const torch::Tensor& t) {
    std::vector<long long> strides(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        strides[i] = t.stride(i);
    }
    return strides;
}

static std::vector<long long> get_shape(const torch::Tensor& t) {
    std::vector<long long> shape(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        shape[i] = t.size(i);
    }
    return shape;
}

// Leaky ReLU wrapper
torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope = 0.01) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    std::vector<long long> shape = get_shape(x);
    std::vector<long long> in_strides = get_strides(x);
    std::vector<long long> out_strides = get_strides(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        x.dim(),
        shape.data(),
        in_strides.data(),
        out_strides.data(),
        static_cast<float>(negative_slope),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after leaky_relu_launcher: ", cudaGetErrorString(err));

    return out;
}

// MaxPool2d wrapper (fixed kernel_size=2, stride=2)
torch::Tensor max_pool2d_forward(torch::Tensor x, int64_t kernel_size = 2, int64_t stride = 2) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_size == 2 && stride == 2, "Only kernel_size=2, stride=2 supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int outH = (H - kernel_size) / stride + 1;
    int outW = (W - kernel_size) / stride + 1;

    auto out = torch::empty({N, C, outH, outW}, x.options());

    std::vector<long long> in_strides = get_strides(x);
    std::vector<long long> out_strides = get_strides(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_pool2d_2x2_stride2_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        in_strides.data(),
        out_strides.data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after max_pool2d_2x2_stride2_launcher: ", cudaGetErrorString(err));

    return out;
}

// Conv2d wrapper (uses custom kernel, no cuDNN)
torch::Tensor conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride = 1,
    int64_t dilation = 1)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor if defined");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (K, C, R, S)");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels must match weight");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);

    // Fixed parameters per contract: stride=1, dilation=1, padding=0
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;
    int dilation_h = 1;
    int dilation_w = 1;

    int outH = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto out = torch::empty({N, K, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        bias.defined() ? 1 : 0,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after conv2d_launcher: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("x"),
          py::arg("negative_slope") = 0.01);

    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward (kernel=2, stride=2)",
          py::arg("x"),
          py::arg("kernel_size") = 2,
          py::arg("stride") = 2);

    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward (custom kernel)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias").none(true),
          py::arg("stride") = 1,
          py::arg("dilation") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
