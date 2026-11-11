#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_avg_pool2d_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_n,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream);

// ---------- custom_abs ----------
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_abs: input must be float32");
    // Ensure input is contiguous for simple elementwise pointer arithmetic.
    // The launcher accepts raw pointer and numel, so contiguous is fine.
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    if (numel == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        numel,
        stream);
    return output;
}

// ---------- custom_avg_pool2d ----------
torch::Tensor custom_avg_pool2d(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_avg_pool2d: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_avg_pool2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_avg_pool2d: input must be 4D (N,C,H,W)");

    // The launcher supports arbitrary strides via explicit stride parameters,
    // so we can pass the original tensor without requiring contiguity.
    // However, @torch::empty_like on non-contiguous input would produce non-contiguous output
    // with same strides, which is acceptable and matches reference behavior.
    // To simplify and improve performance, we make input contiguous if it isn't already.
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t H = x_contig.size(2);
    int64_t W = x_contig.size(3);

    // For contiguous tensors, strides are computed directly.
    int64_t in_stride_n = x_contig.stride(0);
    int64_t in_stride_c = x_contig.stride(1);
    int64_t in_stride_h = x_contig.stride(2);
    int64_t in_stride_w = x_contig.stride(3);
    int64_t out_stride_n = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    if (N == 0 || C == 0 || H == 0 || W == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_avg_pool2d_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);
    return output;
}

// ---------- registration ----------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "custom_abs forward", py::arg("x"));
    m.def("custom_avg_pool2d", &custom_avg_pool2d, "custom_avg_pool2d forward", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
