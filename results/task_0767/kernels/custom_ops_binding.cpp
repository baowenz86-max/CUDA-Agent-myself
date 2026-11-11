#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void softmax_exp_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int64_t input_stride_batch,
    int64_t input_stride_height,
    int64_t input_stride_width,
    int64_t output_stride_batch,
    int64_t output_stride_height,
    int64_t output_stride_width,
    int64_t dim,
    cudaStream_t stream);

extern "C" void softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int64_t input_stride_batch,
    int64_t input_stride_height,
    int64_t input_stride_width,
    int64_t output_stride_batch,
    int64_t output_stride_height,
    int64_t output_stride_width,
    int64_t dim,
    cudaStream_t stream);

extern "C" void exp_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

namespace {

torch::Tensor check_softmax_input(const torch::Tensor& x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    return x.contiguous();
}

} // anonymous namespace

// Wrapper for fused softmax + exp
torch::Tensor softmax_exp_forward(torch::Tensor x, int64_t dim) {
    auto x_contig = check_softmax_input(x, dim);
    auto output = torch::empty_like(x_contig);

    int64_t batch_size = x_contig.size(0);
    int64_t height = x_contig.size(1);
    int64_t width = x_contig.size(2);

    int64_t input_stride_batch = x_contig.stride(0);
    int64_t input_stride_height = x_contig.stride(1);
    int64_t input_stride_width = x_contig.stride(2);

    int64_t output_stride_batch = output.stride(0);
    int64_t output_stride_height = output.stride(1);
    int64_t output_stride_width = output.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_exp_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        height,
        width,
        input_stride_batch,
        input_stride_height,
        input_stride_width,
        output_stride_batch,
        output_stride_height,
        output_stride_width,
        dim,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

// Wrapper for softmax only
torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
    auto x_contig = check_softmax_input(x, dim);
    auto output = torch::empty_like(x_contig);

    int64_t batch_size = x_contig.size(0);
    int64_t height = x_contig.size(1);
    int64_t width = x_contig.size(2);

    int64_t input_stride_batch = x_contig.stride(0);
    int64_t input_stride_height = x_contig.stride(1);
    int64_t input_stride_width = x_contig.stride(2);

    int64_t output_stride_batch = output.stride(0);
    int64_t output_stride_height = output.stride(1);
    int64_t output_stride_width = output.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        height,
        width,
        input_stride_batch,
        input_stride_height,
        input_stride_width,
        output_stride_batch,
        output_stride_height,
        output_stride_width,
        dim,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

// Wrapper for elementwise exp
torch::Tensor exp_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    exp_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("softmax_exp_forward", &softmax_exp_forward,
          "Fused softmax over dim=1 followed by exp",
          pybind11::arg("x"),
          pybind11::arg("dim") = 1);
    m.def("softmax_forward", &softmax_forward,
          "Softmax over dim=1",
          pybind11::arg("x"),
          pybind11::arg("dim") = 1);
    m.def("exp_forward", &exp_forward,
          "Elementwise exp",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
