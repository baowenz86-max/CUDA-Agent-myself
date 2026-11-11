// Binding for custom ops: avg_pool3d_forward and conv3d_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare the average pooling launcher from custom_ops.cu
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    cudaStream_t stream);

// Declare the convolution launcher from custom_ops.cu
extern "C" void conv3d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D, int H, int W,
    int K, int kD, int kH, int kW,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    bool has_bias,
    cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

// Average pooling forward wrapper
// Note: kernel_size parameter is ignored; only kernel_size=2 is supported by the underlying kernel.
torch::Tensor avg_pool3d_forward(torch::Tensor x, int64_t kernel_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(kernel_size == 2, "Only kernel_size=2 is supported");

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int D_out = D / 2;
    int H_out = H / 2;
    int W_out = W / 2;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        stream
    );

    CHECK_CUDA(cudaGetLastError());
    return output;
}

// Conv3d forward wrapper using custom CUDA kernel
// stride is a single int (applied to all dims), padding is a vector of three ints.
torch::Tensor conv3d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    std::vector<int64_t> padding)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "Bias must be CUDA tensor if defined");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D (out_channels, in_channels/groups, kD, kH, kW)");
    TORCH_CHECK(padding.size() == 3, "Padding must be a 3-tuple");

    int N = x.size(0);
    int C_in = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int K_out = weight.size(0);
    int C_in_per_group = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    TORCH_CHECK(C_in == C_in_per_group, "For groups=1, weight in_channels must match input channels");

    int pad_d = static_cast<int>(padding[0]);
    int pad_h = static_cast<int>(padding[1]);
    int pad_w = static_cast<int>(padding[2]);
    int stride_d = static_cast<int>(stride);
    int stride_h = static_cast<int>(stride);
    int stride_w = static_cast<int>(stride);

    int D_out = (D + 2 * pad_d - kD) / stride_d + 1;
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    auto output = torch::empty({N, K_out, D_out, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv3d_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D, H, W,
        K_out, kD, kH, kW,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        bias.defined(),
        stream
    );

    CHECK_CUDA(cudaGetLastError());
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "3D average pooling forward",
          py::arg("x"), py::arg("kernel_size"));
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
