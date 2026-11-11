#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void softmax_dim1_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    cudaStream_t stream);

extern "C" void slice_channels_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t max_channels,
    int64_t C_in,
    int64_t H,
    int64_t W,
    int64_t stride_n_in,
    int64_t stride_c_in,
    int64_t stride_h_in,
    int64_t stride_w_in,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// clamp_forward
// -----------------------------------------------------------------------------
torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "clamp_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clamp_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "clamp_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "clamp_forward: kernel launch failed");
    return output;
}

// -----------------------------------------------------------------------------
// softmax_dim1_forward
// -----------------------------------------------------------------------------
torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "softmax_dim1_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax_dim1_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "softmax_dim1_forward: input must be 4D [N, C, H, W]");
    TORCH_CHECK(x.is_contiguous(), "softmax_dim1_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    if (N == 0 || C == 0 || H == 0 || W == 0) return output;

    // use contiguous strides; .cu assumes arbitrary strides, but we provide contiguous strides
    int64_t stride_n = x.stride(0);
    int64_t stride_c = x.stride(1);
    int64_t stride_h = x.stride(2);
    int64_t stride_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_dim1_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "softmax_dim1_forward: kernel launch failed");
    return output;
}

// -----------------------------------------------------------------------------
// slice_channels_forward
// -----------------------------------------------------------------------------
torch::Tensor slice_channels_forward(torch::Tensor x, int64_t max_channels) {
    TORCH_CHECK(x.is_cuda(), "slice_channels_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "slice_channels_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "slice_channels_forward: input must be 4D [N, C, H, W]");
    TORCH_CHECK(x.is_contiguous(), "slice_channels_forward: input must be contiguous");
    const int64_t C_in = x.size(1);
    TORCH_CHECK(max_channels >= 0 && max_channels <= C_in, "slice_channels_forward: max_channels out of range");

    auto output = torch::empty({x.size(0), max_channels, x.size(2), x.size(3)}, x.options());
    int64_t N = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    if (N == 0 || max_channels == 0 || H == 0 || W == 0) return output;

    int64_t stride_n_in = x.stride(0);
    int64_t stride_c_in = x.stride(1);
    int64_t stride_h_in = x.stride(2);
    int64_t stride_w_in = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    slice_channels_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, max_channels, C_in, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "slice_channels_forward: kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward, "Element-wise clamp",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax over dim 1",
          py::arg("x"));
    m.def("slice_channels_forward", &slice_channels_forward, "Slice first max_channels channels",
          py::arg("x"), py::arg("max_channels"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
