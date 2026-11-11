#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    long long total_elements,
    cudaStream_t stream);

extern "C" void max_pool2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int outH, int outW,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream);

// ReLU forward wrapper
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "relu_forward: input must be float32");
    // No contiguity requirement; kernel uses linear index and works with arbitrary strides (input is contiguous in our case but we support general)
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        (long long)x.numel(),
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_forward: CUDA kernel launch failed");
    return output;
}

// MaxPool2d forward wrapper
torch::Tensor max_pool2d_forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode) {

    TORCH_CHECK(x.is_cuda(), "max_pool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "max_pool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "max_pool2d_forward: input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Convert parameters to int (check reasonable ranges)
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0,
        "max_pool2d_forward: invalid kernel/stride/padding/dilation values");
    int kH = static_cast<int>(kernel_size);
    int kW = static_cast<int>(kernel_size);
    int sH = static_cast<int>(stride);
    int sW = static_cast<int>(stride);
    int pH = static_cast<int>(padding);
    int pW = static_cast<int>(padding);
    int dH = static_cast<int>(dilation);
    int dW = static_cast<int>(dilation);

    // Compute output spatial dimensions (PyTorch MaxPool2d formula)
    int64_t outH_ll, outW_ll;
    if (ceil_mode) {
        outH_ll = (H + 2 * pH - dH * (kH - 1) - 1 + sH - 1) / sH + 1;
        outW_ll = (W + 2 * pW - dW * (kW - 1) - 1 + sW - 1) / sW + 1;
    } else {
        outH_ll = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        outW_ll = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    }
    TORCH_CHECK(outH_ll > 0 && outW_ll > 0, "max_pool2d_forward: computed output size is non-positive");
    int outH = static_cast<int>(outH_ll);
    int outW = static_cast<int>(outW_ll);

    // Allocate output tensor with same options as input
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Get strides as long long
    long long in_stride_n = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_h = x.stride(2);
    long long in_stride_w = x.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool2d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        kH, kW,
        sH, sW,
        pH, pW,
        dH, dW,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "max_pool2d_forward: CUDA kernel launch failed");
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward,
          "ReLU forward (max(x, 0))",
          pybind11::arg("x"));
    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward",
          pybind11::arg("x"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
