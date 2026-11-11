#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launchers from custom_ops.cu
extern "C" void sum_dim1_kernel_launcher(const float* input,
                                        float* output,
                                        int64_t N, int64_t C, int64_t H, int64_t W,
                                        int64_t in_stride_n, int64_t in_stride_c,
                                        int64_t in_stride_h, int64_t in_stride_w,
                                        int64_t out_stride_n, int64_t out_stride_c,
                                        int64_t out_stride_h, int64_t out_stride_w,
                                        cudaStream_t stream);

extern "C" void log10_kernel_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream);

// Helper to ensure tensor is on CUDA and float32
static torch::Tensor ensure_cuda_float(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    if (!x.is_cuda()) {
        // Move to current CUDA device
        x = x.to(torch::kCUDA);
    }
    return x;
}

// Wrapper for sum over dimension 1 with keepdim
torch::Tensor sum_dim1_kernel(torch::Tensor x) {
    x = ensure_cuda_float(x);
    TORCH_CHECK(x.dim() == 4, "sum_dim1_kernel: input must be 4D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output with shape [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, x.options());

    int64_t in_stride_n = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);
    int64_t out_stride_n = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim1_kernel_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim1_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for elementwise log10
torch::Tensor log10_kernel(torch::Tensor x) {
    x = ensure_cuda_float(x);
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log10_kernel_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log10_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1_kernel", &sum_dim1_kernel, "Sum over dim=1 keeping dimension",
          pybind11::arg("x"));
    m.def("log10_kernel", &log10_kernel, "Elementwise log10",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
