#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_neg_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t C, int64_t T, int64_t O,
    int64_t stride_b, int64_t stride_c, int64_t stride_t,
    cudaStream_t stream);

extern "C" void custom_addcmul_tt_launcher(
    const float* bias,
    const float* t1,
    const float* t2,
    float* out,
    int64_t B, int64_t C, int64_t O,
    int64_t t1_stride_b, int64_t t1_stride_c, int64_t t1_stride_o,
    int64_t t2_stride_b, int64_t t2_stride_c, int64_t t2_stride_o,
    cudaStream_t stream);

// ============ PyTorch wrappers ============

torch::Tensor custom_neg(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_neg: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_neg: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "custom_neg: input must have at least one dimension");

    auto out = torch::empty_like(x);
    if (x.numel() == 0) {
        return out;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_neg_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_neg: CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor custom_adaptive_avg_pool1d(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "custom_adaptive_avg_pool1d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_adaptive_avg_pool1d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_adaptive_avg_pool1d: input must be 3D [B, C, T]");
    TORCH_CHECK(output_size > 0, "custom_adaptive_avg_pool1d: output_size must be positive");

    auto B = x.size(0);
    auto C = x.size(1);
    auto T = x.size(2);

    auto out = torch::empty({B, C, output_size}, x.options());
    if (out.numel() == 0) {
        return out;
    }

    // Use strides of input (supports non-contiguous)
    auto stride_b = x.stride(0);
    auto stride_c = x.stride(1);
    auto stride_t = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_adaptive_avg_pool1d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, T, output_size,
        stride_b, stride_c, stride_t,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_adaptive_avg_pool1d: CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor custom_addcmul_tt(torch::Tensor bias, torch::Tensor t1, torch::Tensor t2) {
    TORCH_CHECK(bias.is_cuda(), "custom_addcmul_tt: bias must be a CUDA tensor");
    TORCH_CHECK(t1.is_cuda(), "custom_addcmul_tt: t1 must be a CUDA tensor");
    TORCH_CHECK(t2.is_cuda(), "custom_addcmul_tt: t2 must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "custom_addcmul_tt: bias must be float32");
    TORCH_CHECK(t1.dtype() == torch::kFloat32, "custom_addcmul_tt: t1 must be float32");
    TORCH_CHECK(t2.dtype() == torch::kFloat32, "custom_addcmul_tt: t2 must be float32");
    TORCH_CHECK(t1.dim() == 3, "custom_addcmul_tt: t1 must be 3D [B, C, O]");
    TORCH_CHECK(t2.dim() == 3, "custom_addcmul_tt: t2 must be 3D [B, C, O]");
    TORCH_CHECK(t1.size(0) == t2.size(0) && t1.size(1) == t2.size(1) && t1.size(2) == t2.size(2),
                "custom_addcmul_tt: t1 and t2 must have the same shape");
    TORCH_CHECK(bias.numel() == t1.size(2), "custom_addcmul_tt: bias must have length equal to last dimension (dim 2)");

    auto B = t1.size(0);
    auto C = t1.size(1);
    auto O = t1.size(2);

    auto out = torch::empty_like(t1);
    if (out.numel() == 0) {
        return out;
    }

    // Bias must be contiguous (usually is, but ensure)
    auto bias_contig = bias.contiguous();

    // T1 and T2 can have arbitrary strides; pass them directly
    auto t1_stride_b = t1.stride(0);
    auto t1_stride_c = t1.stride(1);
    auto t1_stride_o = t1.stride(2);
    auto t2_stride_b = t2.stride(0);
    auto t2_stride_c = t2.stride(1);
    auto t2_stride_o = t2.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_addcmul_tt_launcher(
        bias_contig.data_ptr<float>(),
        t1.data_ptr<float>(),
        t2.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, O,
        t1_stride_b, t1_stride_c, t1_stride_o,
        t2_stride_b, t2_stride_c, t2_stride_o,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_addcmul_tt: CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

// ============ Registration ============

void register_custom_ops(pybind11::module& m) {
    m.def("custom_neg", &custom_neg, "Element-wise negation (CUDA)",
          py::arg("x"));
    m.def("custom_adaptive_avg_pool1d", &custom_adaptive_avg_pool1d,
          "Adaptive average pooling 1D (CUDA)",
          py::arg("x"), py::arg("output_size"));
    m.def("custom_addcmul_tt", &custom_addcmul_tt,
          "Addcmul with bias broadcasting over last dimension (CUDA)",
          py::arg("bias"), py::arg("t1"), py::arg("t2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
