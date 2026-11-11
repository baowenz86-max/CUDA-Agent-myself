#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int pad,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long total_elements,
    cudaStream_t stream);

extern "C" void add_scalar_launcher(
    const float* input,
    float* output,
    const float* scalar_ptr,
    long long total_elements,
    cudaStream_t stream);

// reflection_pad2d_forward: x shape [N,C,H,W], padding_size int (symmetric pad)
torch::Tensor reflection_pad2d_forward(torch::Tensor x, int64_t padding_size) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(padding_size >= 0, "padding_size must be non-negative");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int H_out = H + 2 * (int)padding_size;
    int W_out = W + 2 * (int)padding_size;

    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    long long input_stride_n = x.stride(0);
    long long input_stride_c = x.stride(1);
    long long input_stride_h = x.stride(2);
    long long input_stride_w = x.stride(3);
    long long total_elements = (long long)N * C * H_out * W_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        (int)padding_size,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        total_elements,
        stream);

    return output;
}

// add_scalar_forward: x shape [N,C,H,W], scalar shape [1,1,1,1]
torch::Tensor add_scalar_forward(torch::Tensor x, torch::Tensor scalar) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(scalar.is_cuda(), "scalar must be a CUDA tensor");
    TORCH_CHECK(scalar.dtype() == torch::kFloat32, "scalar must be float32");
    TORCH_CHECK(scalar.sizes() == torch::IntArrayRef({1, 1, 1, 1}), "scalar must be shape [1,1,1,1]");
    TORCH_CHECK(scalar.is_contiguous(), "scalar must be contiguous");

    auto output = torch::empty_like(x);

    long long total_elements = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_scalar_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar.data_ptr<float>(),
        total_elements,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2d forward",
          py::arg("x"), py::arg("padding_size"));
    m.def("add_scalar_forward", &add_scalar_forward,
          "Add scalar tensor elementwise",
          py::arg("x"), py::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
