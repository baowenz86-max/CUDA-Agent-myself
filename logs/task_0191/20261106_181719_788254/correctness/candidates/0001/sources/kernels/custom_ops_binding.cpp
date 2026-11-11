#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void instance_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float eps,
    cudaStream_t stream
);

extern "C" void abs_forward_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream
);

// Instance normalization forward wrapper
torch::Tensor instance_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps = 1e-5
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(weight.numel() == C, "weight size must match C");
    TORCH_CHECK(bias.numel() == C, "bias size must match C");

    // Ensure input contiguous for simple stride handling (launcher supports strides, but
    // contiguous ensures coalesced access; we may also pass original strides if desired.
    // To be safe and simple, make contiguous.)
    torch::Tensor x_contig = x.contiguous();

    // Allocate output as contiguous tensor with same shape
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute strides (in elements) for contiguous input and output
    long long in_stride_n = (long long)x_contig.stride(0);
    long long in_stride_c = (long long)x_contig.stride(1);
    long long in_stride_h = (long long)x_contig.stride(2);
    long long in_stride_w = (long long)x_contig.stride(3);
    long long out_stride_n = (long long)y.stride(0);
    long long out_stride_c = (long long)y.stride(1);
    long long out_stride_h = (long long)y.stride(2);
    long long out_stride_w = (long long)y.stride(3);

    // Call launcher
    instance_norm_forward_launcher(
        x_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        static_cast<float>(eps),
        stream
    );

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "instance_norm_forward kernel launch failed");

    return y;
}

// Abs forward wrapper
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    // Ensure contiguous for simple kernel
    torch::Tensor x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    long long numel = static_cast<long long>(x_contig.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        numel,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "abs_forward kernel launch failed");

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "Instance normalization forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps") = 1e-5);
    m.def("abs_forward", &abs_forward,
          "Element-wise absolute value",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
