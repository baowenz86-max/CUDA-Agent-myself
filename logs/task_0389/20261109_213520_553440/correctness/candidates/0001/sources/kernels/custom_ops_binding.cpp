#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm2d_eval_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream);

extern "C" void softmax_channel_launcher(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream);

// Helper to get strides for 4D tensor
static void get_strides(const torch::Tensor& t, int64_t& stride_n, int64_t& stride_c, int64_t& stride_h, int64_t& stride_w) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor");
    auto strides = t.strides();
    stride_n = strides[0];
    stride_c = strides[1];
    stride_h = strides[2];
    stride_w = strides[3];
}

torch::Tensor batch_norm2d_eval(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    // Validate x
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Validate parameter tensors
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32 && running_mean.numel() == C, "running_mean must be CUDA float32 of size C");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32 && running_var.numel() == C, "running_var must be CUDA float32 of size C");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda() && weight->dtype() == torch::kFloat32 && weight->numel() == C, "weight must be CUDA float32 of size C");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->dtype() == torch::kFloat32 && bias->numel() == C, "bias must be CUDA float32 of size C");
        bias_ptr = bias->data_ptr<float>();
    }

    // Allocate output with same strides
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t stride_x_n, stride_x_c, stride_x_h, stride_x_w;
    int64_t stride_y_n, stride_y_c, stride_y_h, stride_y_w;
    get_strides(x, stride_x_n, stride_x_c, stride_x_h, stride_x_w);
    get_strides(y, stride_y_n, stride_y_c, stride_y_h, stride_y_w);

    batch_norm2d_eval_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

torch::Tensor softmax_channel(torch::Tensor x)
{
    // Validate x
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t stride_x_n, stride_x_c, stride_x_h, stride_x_w;
    int64_t stride_y_n, stride_y_c, stride_y_h, stride_y_w;
    get_strides(x, stride_x_n, stride_x_c, stride_x_h, stride_x_w);
    get_strides(y, stride_y_n, stride_y_c, stride_y_h, stride_y_w);

    softmax_channel_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_eval", &batch_norm2d_eval,
          "BatchNorm2d inference (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("softmax_channel", &softmax_channel,
          "Softmax over channel dimension (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);