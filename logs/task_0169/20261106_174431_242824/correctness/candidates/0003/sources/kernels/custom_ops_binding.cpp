#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm2d_forward_launcher(
    const float* x,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    bool training,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void hardsigmoid_forward_launcher(
    const float* x,
    float* output,
    float slope,
    long long numel,
    cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    const float* x,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

// PyTorch wrapper for batch_norm2d_forward
torch::Tensor batch_norm2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    bool training) {
    // Check training flag
    TORCH_CHECK(!training, "batch_norm2d_forward only supports eval mode (training=False)");

    // Check x
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    // Check parameter tensors
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(running_mean.numel() == C, "running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == C, "running_var size mismatch");
    if (weight.defined()) TORCH_CHECK(weight.numel() == C, "weight size mismatch");
    if (bias.defined()) TORCH_CHECK(bias.numel() == C, "bias size mismatch");

    // Allocate output with same shape and strides as input
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides (long long)
    long long x_stride_n = x.stride(0);
    long long x_stride_c = x.stride(1);
    long long x_stride_h = x.stride(2);
    long long x_stride_w = x.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    // Call launcher
    batch_norm2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        training,
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

// PyTorch wrapper for hardsigmoid_forward
torch::Tensor hardsigmoid_forward(torch::Tensor x, double slope) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto output = torch::empty_like(x);
    long long numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardsigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(slope),
        numel,
        stream
    );

    return output;
}

// PyTorch wrapper for add_bias_forward
torch::Tensor add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (C,1,1)");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias shape must be [C,1,1]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(bias.size(0) == C, "bias channel dimension mismatch");

    // Ensure bias is contiguous
    auto bias_contig = bias.contiguous();

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long x_stride_n = x.stride(0);
    long long x_stride_c = x.stride(1);
    long long x_stride_h = x.stride(2);
    long long x_stride_w = x.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    add_bias_forward_launcher(
        x.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_forward", &batch_norm2d_forward,
          "BatchNorm2d forward (eval mode)",
          py::arg("x"), py::arg("weight") = py::none(), py::arg("bias") = py::none(),
          py::arg("running_mean"), py::arg("running_var"),
          py::arg("eps"), py::arg("training"));
    m.def("hardsigmoid_forward", &hardsigmoid_forward,
          "Hardsigmoid forward",
          py::arg("x"), py::arg("slope"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);