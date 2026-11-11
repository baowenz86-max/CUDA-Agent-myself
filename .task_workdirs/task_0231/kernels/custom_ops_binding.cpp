#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations matching custom_ops.cu
extern "C" void hardtanh_clamp_launcher(
    const float* input,
    float* output,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void mean_channel_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream);

namespace {

torch::Tensor hardtanh_clamp_forward(
    torch::Tensor input,
    double min_val = -1.0,
    double max_val = 1.0) {
    TORCH_CHECK(input.is_cuda(), "hardtanh_clamp: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "hardtanh_clamp: input must be float32");

    // Ensure contiguous for simple pointer-based kernel
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int64_t numel = input_contig.numel();
    if (numel == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardtanh_clamp_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardtanh_clamp kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor mean_channel_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "mean_channel: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "mean_channel: input must be float32");
    TORCH_CHECK(input.dim() == 4, "mean_channel: input must be 4D (N,C,H,W)");

    // Ensure contiguous
    auto input_contig = input.contiguous();
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    TORCH_CHECK(C > 0, "mean_channel: channel dimension must be positive");

    // Allocate output shape [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, input_contig.options());

    if (N == 0 || H == 0 || W == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_channel_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_channel kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh_clamp", &hardtanh_clamp_forward,
          "Elementwise hardtanh clamp",
          py::arg("input"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);
    m.def("mean_channel", &mean_channel_forward,
          "Mean over channel dimension (dim=1, keepdim=True)",
          py::arg("input"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
