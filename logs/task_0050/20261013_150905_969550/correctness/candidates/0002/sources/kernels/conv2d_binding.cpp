#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu
extern "C" void conv2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream);

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    c10::optional<torch::Tensor> bias_opt) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias_opt.has_value() == false || (bias_opt.value().defined() && bias_opt.value().is_cuda() && bias_opt.value().dtype() == torch::kFloat32), "bias must be None or a CUDA float32 tensor");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t K = weight_contig.size(0);
    int64_t C_per_group = weight_contig.size(1);
    int64_t R = weight_contig.size(2);
    int64_t S = weight_contig.size(3);

    TORCH_CHECK(C == C_per_group * groups, "input channels must equal weight channels * groups");

    int64_t H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, K, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(K),
        static_cast<int>(R),
        static_cast<int>(S),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(groups),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv2d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_conv2d(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "2D convolution forward using custom CUDA kernel",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("stride_h"),
          pybind11::arg("stride_w"),
          pybind11::arg("padding_h"),
          pybind11::arg("padding_w"),
          pybind11::arg("dilation_h"),
          pybind11::arg("dilation_w"),
          pybind11::arg("groups") = 1,
          pybind11::arg("bias") = pybind11::none());
}

REGISTER_BINDING(conv2d, register_conv2d)
