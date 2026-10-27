// Binding file for custom ops without cuDNN dependency.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream);

extern "C" void bias_add_forward_launcher(
    const float* x,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream);

extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int kH, int kW,
    int outH, int outW,
    int use_bias,
    cudaStream_t stream);

// Helper: check tensor is CUDA float32
torch::Tensor triu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "triu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "triu_forward: input must be float32");
    TORCH_CHECK(x.dim() >= 2, "triu_forward: input must have at least 2 dimensions");

    // Output should preserve shape and strides to ensure no layout change
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t H = x.size(-2);
    int64_t W = x.size(-1);
    int64_t N = 1, C = 1;
    if (x.dim() == 4) {
        N = x.size(0);
        C = x.size(1);
    } else if (x.dim() == 2) {
        N = x.size(0);
        C = 1;
    } else {
        N = x.numel() / (H * W);
        C = 1;
    }

    long long stride_batch, stride_channel, stride_h, stride_w;
    if (x.dim() == 4) {
        stride_batch = x.stride(0);
        stride_channel = x.stride(1);
        stride_h = x.stride(2);
        stride_w = x.stride(3);
    } else if (x.dim() == 2) {
        stride_batch = 0;
        stride_channel = 0;
        stride_h = x.stride(0);
        stride_w = x.stride(1);
    } else {
        stride_batch = H * W;
        stride_channel = H * W;
        stride_h = W;
        stride_w = 1;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy input to output if different pointers.
    if (output.data_ptr<float>() != x.data_ptr<float>()) {
        cudaMemcpyAsync(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    triu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stride_batch, stride_channel, stride_h, stride_w,
        stream
    );

    return output;
}

torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "bias_add_forward: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias_add_forward: bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "bias_add_forward: x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias_add_forward: bias must be float32");
    TORCH_CHECK(x.dim() == 4, "bias_add_forward: x must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.numel() == x.size(1), "bias_add_forward: bias size must match C");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));

    long long stride_batch = x.stride(0);
    long long stride_channel = x.stride(1);
    long long stride_h = x.stride(2);
    long long stride_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_forward_launcher(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_batch, stride_channel, stride_h, stride_w,
        stream
    );

    return output;
}

// Convolution wrapper using custom direct kernel (no cuDNN).
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "conv2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d_forward: weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d_forward: weight must be float32");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "conv2d_forward: bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "conv2d_forward: bias must be float32");
        TORCH_CHECK(bias->numel() == weight.size(0), "conv2d_forward: bias size must match out_channels");
    }
    TORCH_CHECK(input.dim() == 4, "conv2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "conv2d_forward: weight must be 4D (out_channels, in_channels/groups, kH, kW)");
    TORCH_CHECK(stride == 1 && padding == 0 && dilation == 1 && groups == 1, "conv2d_forward: only stride=1, padding=0, dilation=1, groups=1 supported");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));
    int K = static_cast<int>(weight.size(0));
    int kH = static_cast<int>(weight.size(2));
    int kW = static_cast<int>(weight.size(3));

    int outH = (H - kH) + 1;
    int outW = (W - kW) + 1;

    auto output = torch::empty({N, K, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    int use_bias = 0;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
        use_bias = 1;
    }

    conv2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W, K, kH, kW, outH, outW,
        use_bias,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Apply upper triangular mask to last two dimensions",
          py::arg("x"));
    m.def("conv2d_forward", &conv2d_forward, "2D convolution via custom kernel",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("bias_add_forward", &bias_add_forward, "Add bias with broadcasting",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
