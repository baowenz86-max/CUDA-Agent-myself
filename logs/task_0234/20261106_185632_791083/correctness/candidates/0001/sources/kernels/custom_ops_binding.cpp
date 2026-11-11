// custom_ops_binding.cpp
// PyTorch bindings for custom ops implemented in custom_ops.cu.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu (extern "C")
extern "C" void batchnorm_eval_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    cudaStream_t stream);

extern "C" void abs_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    cudaStream_t stream);

// Helper: ensure tensor is contiguous float32 CUDA tensor; return new tensor if needed.
static torch::Tensor ensure_contiguous(const torch::Tensor& t) {
    if (t.is_contiguous()) {
        return t;
    }
    return t.contiguous();
}

torch::Tensor batchnorm_eval_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps = 1e-5)
{
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && running_mean.dim() == 1 && running_var.dim() == 1, "parameter tensors must be 1D");
    TORCH_CHECK(weight.size(0) == x.size(1) && bias.size(0) == x.size(1) && running_mean.size(0) == x.size(1) && running_var.size(0) == x.size(1), "channel dimension mismatch");

    // Make input contiguous (launcher expects strides, but contiguous is faster)
    auto x_contig = ensure_contiguous(x);
    // Parameters are guaranteed contiguous? Typically they are, but enforce.
    auto weight_contig = ensure_contiguous(weight);
    auto bias_contig = ensure_contiguous(bias);
    auto running_mean_contig = ensure_contiguous(running_mean);
    auto running_var_contig = ensure_contiguous(running_var);

    int N = static_cast<int>(x_contig.size(0));
    int C = static_cast<int>(x_contig.size(1));
    int H = static_cast<int>(x_contig.size(2));
    int W = static_cast<int>(x_contig.size(3));

    // Allocate output with same shape as x, contiguous (matching y strides)
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides for input and output (contiguous: y has same strides as x_contig)
    long long x_stride_n = x_contig.stride(0);
    long long x_stride_c = x_contig.stride(1);
    long long x_stride_h = x_contig.stride(2);
    long long x_stride_w = x_contig.stride(3);
    long long y_stride_n = y.stride(0);
    long long y_stride_c = y.stride(1);
    long long y_stride_h = y.stride(2);
    long long y_stride_w = y.stride(3);

    batchnorm_eval_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batchnorm_eval_launcher failed: ", cudaGetErrorString(err));

    return y;
}

torch::Tensor abs_forward(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    auto x_contig = ensure_contiguous(x);
    auto y = torch::empty_like(x_contig);

    int N = static_cast<int>(x_contig.size(0));
    int C = static_cast<int>(x_contig.size(1));
    int H = static_cast<int>(x_contig.size(2));
    int W = static_cast<int>(x_contig.size(3));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long x_stride_n = x_contig.stride(0);
    long long x_stride_c = x_contig.stride(1);
    long long x_stride_h = x_contig.stride(2);
    long long x_stride_w = x_contig.stride(3);
    long long y_stride_n = y.stride(0);
    long long y_stride_c = y.stride(1);
    long long y_stride_h = y.stride(2);
    long long y_stride_w = y.stride(3);

    abs_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "abs_launcher failed: ", cudaGetErrorString(err));

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm_eval_forward", &batchnorm_eval_forward,
          "BatchNorm2d evaluation forward (x, weight, bias, running_mean, running_var, eps=1e-5)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps") = 1e-5);
    m.def("abs_forward", &abs_forward,
          "Elementwise absolute value forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
