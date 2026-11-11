// Binding for mean_dim_last and softplus custom CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void mean_dim_last_launcher(
    float* out,
    const float* x,
    int B,
    int C,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream
);

extern "C" void softplus_launcher(
    float* out,
    const float* x,
    long long numel,
    long long x_stride0,
    long long x_stride1,
    long long x_stride2,
    long long x_stride3,
    long long out_stride0,
    long long out_stride1,
    long long out_stride2,
    long long out_stride3,
    int ndim,
    cudaStream_t stream
);

// Helper to check for CUDA errors after kernel launch
static void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " kernel launch failed: ", cudaGetErrorString(err));
}

// PyTorch wrapper for mean_dim_last
torch::Tensor mean_dim_last(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mean_dim_last: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim_last: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "mean_dim_last: input must have at least 1 dimension");

    // Treat input as 4D: [B, C, H, W] with implicit leading singleton dims if needed
    int extra_dims = (x.dim() < 4) ? (4 - x.dim()) : 0;
    int B = extra_dims > 0 ? 1 : x.size(0);
    int C = extra_dims > 2 ? 1 : x.size(1);
    int H = extra_dims > 1 ? 1 : x.size(2);
    int W = x.size(-1);

    // For dim < 4, adjust strides: missing dimensions have stride 0 or same as W? For broadcasting, we only use valid strides.
    // To simplify, we use actual strides if dims exist, 0 otherwise; but launcher would add zero offset, which is fine.
    long long x_stride_b = (extra_dims > 0) ? 0 : x.stride(0);
    long long x_stride_c = (extra_dims > 2) ? 0 : x.stride(1);
    long long x_stride_h = (extra_dims > 1) ? 0 : x.stride(2);
    long long x_stride_w = x.stride(-1);

    // Output shape: [B, C, H, 1]
    auto out = torch::empty({B, C, H, 1}, x.options());
    long long out_stride_b = (extra_dims > 0) ? 0 : out.stride(0);
    long long out_stride_c = (extra_dims > 2) ? 0 : out.stride(1);
    long long out_stride_h = (extra_dims > 1) ? 0 : out.stride(2);
    long long out_stride_w = out.stride(-1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim_last_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        B, C, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stream
    );
    check_cuda_error("mean_dim_last");
    return out;
}

// PyTorch wrapper for softplus
torch::Tensor softplus(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "softplus: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softplus: input must be float32");

    auto out = torch::empty_like(x);
    long long numel = x.numel();
    int ndim = x.dim();

    // Prepare strides for up to 4 dimensions; for fewer, use 0 for unused dims
    long long x_stride[4] = {0, 0, 0, 0};
    long long out_stride[4] = {0, 0, 0, 0};
    for (int d = 0; d < ndim && d < 4; ++d) {
        x_stride[d] = x.stride(d);
        out_stride[d] = out.stride(d);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        x_stride[0], x_stride[1], x_stride[2], x_stride[3],
        out_stride[0], out_stride[1], out_stride[2], out_stride[3],
        ndim,
        stream
    );
    check_cuda_error("softplus");
    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim_last", &mean_dim_last, "Compute mean along the last dimension, keeping dimension (float32 CUDA)");
    m.def("softplus", &softplus, "Apply softplus element-wise (float32 CUDA)");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
