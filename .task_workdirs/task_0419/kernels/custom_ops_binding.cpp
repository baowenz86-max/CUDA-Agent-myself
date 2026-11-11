#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    int64_t out_n,
    int64_t out_c,
    int64_t out_h,
    int64_t out_w,
    int64_t in_n,
    int64_t in_c,
    int64_t in_h,
    int64_t in_w,
    cudaStream_t stream
);

extern "C" void log_softmax_forward_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    int64_t out_n,
    int64_t out_c,
    int64_t out_h,
    int64_t out_w,
    int64_t in_n,
    int64_t in_c,
    int64_t in_h,
    int64_t in_w,
    int64_t dim,
    cudaStream_t stream
);

torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    // Output tensor with same shape and strides (supports strides)
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t n = x.size(0);
    int64_t c = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    int64_t out_n = output.stride(0);
    int64_t out_c = output.stride(1);
    int64_t out_h = output.stride(2);
    int64_t out_w = output.stride(3);
    int64_t in_n = x.stride(0);
    int64_t in_c = x.stride(1);
    int64_t in_h = x.stride(2);
    int64_t in_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        n, c, h, w,
        out_n, out_c, out_h, out_w,
        in_n, in_c, in_h, in_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor log_softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t n = x.size(0);
    int64_t c = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    int64_t out_n = output.stride(0);
    int64_t out_c = output.stride(1);
    int64_t out_h = output.stride(2);
    int64_t out_w = output.stride(3);
    int64_t in_n = x.stride(0);
    int64_t in_c = x.stride(1);
    int64_t in_h = x.stride(2);
    int64_t in_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        n, c, h, w,
        out_n, out_c, out_h, out_w,
        in_n, in_c, in_h, in_w,
        dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, py::arg("x"));
    m.def("log_softmax_forward", &log_softmax_forward, py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
