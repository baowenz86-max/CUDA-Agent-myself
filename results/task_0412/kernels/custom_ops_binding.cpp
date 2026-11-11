// Bindings for batch_norm_eval and avg_pool2d custom CUDA ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void batch_norm_eval_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t C,
    int64_t HW,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(
    const float* x,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t H_out,
    int64_t W_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t total_output_elements,
    cudaStream_t stream);

// Helper to get CUDA stream
cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// batch_norm_eval forward
torch::Tensor batch_norm_eval(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // Check dimensions
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;
    int64_t total = N * C * HW;

    TORCH_CHECK(running_mean.numel() == C, "running_mean size must equal channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must equal channels");

    // Optional weight/bias checks
    const float* weight_ptr = nullptr;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.numel() == C, "weight size must equal channels");
        weight_ptr = w.data_ptr<float>();
    }
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.numel() == C, "bias size must equal channels");
        bias_ptr = b.data_ptr<float>();
    }

    // Allocate output
    auto y = torch::empty_like(x);

    // Get stream and launch
    cudaStream_t stream = getCurrentStream();
    batch_norm_eval_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        C,
        HW,
        total,
        stream
    );

    return y;
}

// avg_pool2d forward
torch::Tensor avg_pool2d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output dimensions
    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output size");

    int64_t total_output = N * C * H_out * W_out;

    // Allocate output
    auto y = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = getCurrentStream();
    avg_pool2d_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding,
        total_output,
        stream
    );

    return y;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_eval", &batch_norm_eval,
          "BatchNorm2d eval mode custom CUDA implementation",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("avg_pool2d", &avg_pool2d,
          "AvgPool2d custom CUDA implementation",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
