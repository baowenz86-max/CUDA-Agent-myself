#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batchnorm2d_eval_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream
);

extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    cudaStream_t stream
);

// Helper: return current CUDA stream
static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// BatchNorm2d eval forward
// Inputs: x [N,C,H,W] (float32, CUDA, may be non-contiguous), weight/bias/running_mean/running_var [C] (float32, CUDA, contiguous), eps (float, optional)
// Output: contiguous CUDA tensor of same shape as x
torch::Tensor batchnorm2d_eval_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps = 1e-05
) {
    // Check types and devices
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.scalar_type() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.scalar_type() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must be 1D of size C");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must be 1D of size C");
    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == C, "running_mean must be 1D of size C");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == C, "running_var must be 1D of size C");

    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // x may be non-contiguous; we support arbitrary strides
    // Output tensor: same shape as x, contiguous
    auto out = torch::empty_like(x);
    // Ensure out is contiguous (empty_like preserves memory format, but x might be non-contiguous; we force contiguous)
    out = out.contiguous();

    // Get strides (in elements)
    auto x_strides = x.strides();
    auto out_strides = out.strides();
    // Convert to long long for launcher
    long long x_stride_n = x_strides[0];
    long long x_stride_c = x_strides[1];
    long long x_stride_h = x_strides[2];
    long long x_stride_w = x_strides[3];
    long long out_stride_n = out_strides[0];
    long long out_stride_c = out_strides[1];
    long long out_stride_h = out_strides[2];
    long long out_stride_w = out_strides[3];

    cudaStream_t stream = get_current_stream();

    batchnorm2d_eval_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        out.data_ptr<float>(),
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "batchnorm2d_eval_forward kernel launch failed");

    return out;
}

// Sum over dimension 1 with keepdim=True
// Input: x [N,C,H,W] (float32, CUDA, may be non-contiguous), dim (int)
// Output: contiguous CUDA tensor of shape [N,1,H,W]
torch::Tensor sum_dim_keepdim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape: [N,1,H,W], contiguous
    auto out = torch::empty({N, 1, H, W}, x.options());

    auto x_strides = x.strides();
    long long x_stride_n = x_strides[0];
    long long x_stride_c = x_strides[1];
    long long x_stride_h = x_strides[2];
    long long x_stride_w = x_strides[3];

    cudaStream_t stream = get_current_stream();

    sum_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sum_dim_keepdim_forward kernel launch failed");

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm2d_eval_forward", &batchnorm2d_eval_forward,
          "BatchNorm2d eval forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps") = 1e-05);

    m.def("sum_dim_keepdim_forward", &sum_dim_keepdim_forward,
          "Sum over dim with keepdim=True",
          pybind11::arg("x"),
          pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
