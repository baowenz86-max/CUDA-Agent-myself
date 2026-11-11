#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    float* mean_buf,
    float* var_buf,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    float eps,
    cudaStream_t stream);

extern "C" void softmax_channel_forward_launcher(
    float* out,
    const float* x,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream);

// GroupNorm forward wrapper
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Strides (long long for 64-bit indexing)
    long long stride_n = static_cast<long long>(x.stride(0));
    long long stride_c = static_cast<long long>(x.stride(1));
    long long stride_h = static_cast<long long>(x.stride(2));
    long long stride_w = static_cast<long long>(x.stride(3));

    // Allocate output with same shape and strides as input
    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Allocate temporary buffers for mean and variance, size N*C
    auto options = x.options();
    auto mean_buf = torch::empty({N * C}, options);
    auto var_buf = torch::empty({N * C}, options);

    // Get weight/bias data pointers (may be null if not provided)
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.defined() && weight.numel() > 0) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.numel() == C, "weight size must match C");
        weight_ptr = weight.data_ptr<float>();
    }
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C, "bias size must match C");
        bias_ptr = bias.data_ptr<float>();
    }

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        mean_buf.data_ptr<float>(),
        var_buf.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stride_n, stride_c, stride_h, stride_w,
        static_cast<float>(eps),
        stream
    );

    return out;
}

// Softmax over channel dimension wrapper
torch::Tensor softmax_channel_forward(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor [N, C, H, W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    long long stride_n = static_cast<long long>(x.stride(0));
    long long stride_c = static_cast<long long>(x.stride(1));
    long long stride_h = static_cast<long long>(x.stride(2));
    long long stride_w = static_cast<long long>(x.stride(3));

    // Allocate output with same shape and strides as input
    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_channel_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stride_n, stride_c, stride_h, stride_w,
        stream
    );

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));

    m.def("softmax_channel_forward", &softmax_channel_forward,
          "Softmax over channel dimension (CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
