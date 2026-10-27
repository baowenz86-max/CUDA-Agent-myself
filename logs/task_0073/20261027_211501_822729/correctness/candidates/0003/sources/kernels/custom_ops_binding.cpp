#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void tril_forward_launcher(
    const float* input,
    float* output,
    long long total_elements,
    int H,
    int W,
    int diagonal,
    long long stride_nc,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

extern "C" void mean_forward_launcher(
    const float* input,
    float* output,
    int H,
    int W,
    long long outer_size,
    long long stride_nc,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// tril_forward binding
// -----------------------------------------------------------------------------
torch::Tensor tril_forward(torch::Tensor x, int64_t diagonal) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");

    int64_t H = x.size(-2);
    int64_t W = x.size(-1);
    TORCH_CHECK(H > 0 && W > 0, "Last two dimensions must be positive");

    int64_t outer_size = 1;
    for (int64_t i = 0; i < x.dim() - 2; ++i) {
        outer_size *= x.size(i);
    }

    int64_t stride_nc = (x.dim() >= 3) ? x.stride(-3) : H * W;
    int64_t stride_h = x.stride(-2);
    int64_t stride_w = x.stride(-1);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tril_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        (int)H,
        (int)W,
        (int)diagonal,
        stride_nc,
        stride_h,
        stride_w,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// mean_forward binding
// -----------------------------------------------------------------------------
torch::Tensor mean_forward(torch::Tensor x, std::vector<int64_t> dims, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");

    TORCH_CHECK(dims.size() == 2, "Expected dims to be a list of length 2");
    TORCH_CHECK(dims[0] == 2 && dims[1] == 3, "Only dims=[2,3] supported");
    TORCH_CHECK(keepdim, "keepdim must be true for this operation");

    int64_t H = x.size(-2);
    int64_t W = x.size(-1);
    TORCH_CHECK(H > 0 && W > 0, "Last two dimensions must be positive");

    int64_t outer_size = 1;
    for (int64_t i = 0; i < x.dim() - 2; ++i) {
        outer_size *= x.size(i);
    }

    int64_t stride_nc = (x.dim() >= 3) ? x.stride(-3) : H * W;
    int64_t stride_h = x.stride(-2);
    int64_t stride_w = x.stride(-1);

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape[out_shape.size() - 2] = 1;
    out_shape[out_shape.size() - 1] = 1;
    auto output = torch::empty(out_shape, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)H,
        (int)W,
        outer_size,
        stride_nc,
        stride_h,
        stride_w,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("tril_forward", &tril_forward, "Apply torch.tril on last two dims",
          py::arg("x"), py::arg("diagonal"));
    m.def("mean_forward", &mean_forward, "Compute mean over dims (2,3) with keepdim",
          py::arg("x"), py::arg("dims"), py::arg("keepdim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
