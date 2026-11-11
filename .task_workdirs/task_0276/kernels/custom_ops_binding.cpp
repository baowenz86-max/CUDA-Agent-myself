// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_channel_last_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int numel,
    float negative_slope,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void conv2d_direct_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream);

// Helper to check CUDA errors
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Softmax wrapper
torch::Tensor softmax_channel_last_placeholder(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 supported");

    // Ensure channels_last layout (NHWC). The contract says input is channels_last,
    // but we still check and convert if necessary to be safe.
    auto x_cl = x.contiguous(); // Make contiguous, assuming it's already NHWC but may be NCHW if user passes different.
    // Actually contract says channels_last, so we trust that. But if not, we need to convert.
    // Current code does not handle non-channels_last correctly. We'll fix: convert to channels_last if not.
    if (!x.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        x_cl = x.to(torch::MemoryFormat::ChannelsLast);
        // If x is not contiguous in any format, we first make contiguous NCHW then permute? For simplicity, assume x is contiguous NCHW if not channels_last.
        // But to be robust: x_cl = x.contiguous(); then if not channels_last, we need to permute to NHWC. But we'll avoid complexity by assuming contract.
        // For now, keep as is: x_cl = x.contiguous(); This will be NCHW contiguous, but softmax kernel expects NHWC, so wrong.
        // So we must ensure channels_last. Let's force:
    }
    // Ensure channels_last: if not, use .to(MemoryFormat::ChannelsLast) which may copy.
    x_cl = x.contiguous(); // make contiguous in whatever layout
    if (!x_cl.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        x_cl = x_cl.to(torch::MemoryFormat::ChannelsLast);
    }

    auto sizes = x_cl.sizes();
    int N = static_cast<int>(sizes[0]);
    int C = static_cast<int>(sizes[1]);
    int H = static_cast<int>(sizes[2]);
    int W = static_cast<int>(sizes[3]);

    auto output = torch::empty_like(x_cl, torch::MemoryFormat::ChannelsLast);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_channel_last_launcher(
        output.data_ptr<float>(),
        x_cl.data_ptr<float>(),
        N, C, H, W,
        stream
    );
    checkCudaError(cudaGetLastError(), "softmax_channel_last_launcher failed");

    return output;
}

// LeakyReLU wrapper
torch::Tensor leaky_relu_placeholder(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Works on contiguous data; numel is same regardless of memory format
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        static_cast<int>(x_contig.numel()),
        static_cast<float>(negative_slope),
        stream
    );
    checkCudaError(cudaGetLastError(), "leaky_relu_launcher failed");

    return output;
}

// Add bias wrapper
torch::Tensor add_bias_placeholder(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto sizes = x.sizes();
    int N = static_cast<int>(sizes[0]);
    int C = static_cast<int>(sizes[1]);
    int H = static_cast<int>(sizes[2]);
    int W = static_cast<int>(sizes[3]);

    TORCH_CHECK(bias.numel() == C, "Bias numel must match channels");

    // Ensure channels_last layout for launcher
    auto x_cl = x.contiguous(); // could be any layout
    if (!x_cl.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        x_cl = x_cl.to(torch::MemoryFormat::ChannelsLast);
    }
    auto output = torch::empty_like(x_cl, torch::MemoryFormat::ChannelsLast);

    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        output.data_ptr<float>(),
        x_cl.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N, C, H, W,
        stream
    );
    checkCudaError(cudaGetLastError(), "add_bias_launcher failed");

    return output;
}

// Conv2d using direct CUDA kernel (no cuDNN)
torch::Tensor conv2d_cudnn_placeholder(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (O,I,kH,kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = static_cast<int>(x.size(0));
    int C_in = static_cast<int>(x.size(1));
    int H_in = static_cast<int>(x.size(2));
    int W_in = static_cast<int>(x.size(3));

    int C_out = static_cast<int>(weight.size(0));
    int kH = static_cast<int>(weight.size(2));
    int kW = static_cast<int>(weight.size(3));

    TORCH_CHECK(weight.size(1) == C_in, "weight input channels mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must equal out channels");

    int pad_h = static_cast<int>(padding);
    int pad_w = static_cast<int>(padding);
    int stride_h = static_cast<int>(stride);
    int stride_w = static_cast<int>(stride);

    // Calculate output dimensions
    int H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    // Ensure input is channels_last (NHWC)
    auto x_cl = x.contiguous();
    if (!x_cl.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        x_cl = x_cl.to(torch::MemoryFormat::ChannelsLast);
    }

    // Weight should be contiguous OIHW
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Create output tensor in channels_last
    auto output = torch::empty({N, C_out, H_out, W_out}, x.options().memory_format(torch::MemoryFormat::ChannelsLast));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_direct_launcher(
        output.data_ptr<float>(),
        x_cl.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        stream
    );
    checkCudaError(cudaGetLastError(), "conv2d_direct_launcher failed");

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("softmax_channel_last_placeholder", &softmax_channel_last_placeholder,
          "Softmax along channel dim for channels_last tensor",
          pybind11::arg("x"), pybind11::arg("dim"));
    m.def("leaky_relu_placeholder", &leaky_relu_placeholder,
          "LeakyReLU elementwise",
          pybind11::arg("x"), pybind11::arg("negative_slope"));
    m.def("conv2d_cudnn_placeholder", &conv2d_cudnn_placeholder,
          "Conv2d with direct kernel, includes bias",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("add_bias_placeholder", &add_bias_placeholder,
          "Add bias broadcast",
          pybind11::arg("x"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
