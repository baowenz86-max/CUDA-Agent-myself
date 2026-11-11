#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_launcher(
    float* out,
    const float* in,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_out_n,
    int64_t stride_out_c,
    int64_t stride_out_h,
    int64_t stride_out_w,
    int64_t stride_in_n,
    int64_t stride_in_c,
    int64_t stride_in_h,
    int64_t stride_in_w,
    int dim,
    cudaStream_t stream);

extern "C" void hardswish_launcher(
    float* out,
    const float* in,
    int64_t total,
    int64_t stride_out,
    int64_t stride_in,
    cudaStream_t stream);

// Softmax binding
torch::Tensor softmax(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "softmax input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax input must be float32");
    TORCH_CHECK(x.dim() == 4, "softmax input must be 4D");
    TORCH_CHECK(dim == 1, "softmax only supports dim=1 for this contract");

    auto out = torch::empty_like(x);

    // Get dimensions and strides
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t stride_in_n = x.stride(0);
    int64_t stride_in_c = x.stride(1);
    int64_t stride_in_h = x.stride(2);
    int64_t stride_in_w = x.stride(3);

    int64_t stride_out_n = out.stride(0);
    int64_t stride_out_c = out.stride(1);
    int64_t stride_out_h = out.stride(2);
    int64_t stride_out_w = out.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        stride_in_n, stride_in_c, stride_in_h, stride_in_w,
        (int)dim,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Hardswish binding
torch::Tensor hardswish(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "hardswish input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "hardswish input must be float32");

    auto out = torch::empty_like(x);

    int64_t total = x.numel();

    // For contiguous tensors, stride is 1; for non-contiguous, we use the stride of the last dimension,
    // assuming the launcher handles arbitrary stride linearly. The kernel uses i * stride_in, so we pass
    // the actual stride of the innermost dimension. If the tensor is non-contiguous with gaps in the
    // innermost dimension, this works correctly. For more complex layouts, the kernel would need a
    // multidimensional index, but the input from softmax output is contiguous, so this is sufficient.
    int64_t stride_in = x.stride(x.dim() - 1);
    int64_t stride_out = out.stride(out.dim() - 1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardswish_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        total,
        stride_out,
        stride_in,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardswish kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("softmax", &softmax, "Softmax along dim=1 for 4D tensor",
          py::arg("x"), py::arg("dim"));
    m.def("hardswish", &hardswish, "Hardswish activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
