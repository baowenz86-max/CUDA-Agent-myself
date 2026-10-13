#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    int output_size,
    cudaStream_t stream);

extern "C" void conv_transpose2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding, int output_padding,
    cudaStream_t stream);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// PyTorch wrapper for adaptive_avg_pool1d custom op
torch::Tensor adaptive_avg_pool1d(torch::Tensor input, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "adaptive_avg_pool1d: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "adaptive_avg_pool1d: input must be float32");
    TORCH_CHECK(input.dim() == 3, "adaptive_avg_pool1d: input must be 3D (N,C,L)");
    TORCH_CHECK(output_size > 0, "adaptive_avg_pool1d: output_size must be positive");

    auto input_contig = input.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int L = input_contig.size(2);

    auto output = torch::empty({N, C, output_size}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool1d_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        L,
        static_cast<int>(output_size),
        stream
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// PyTorch wrapper for conv_transpose2d using custom CUDA kernel
torch::Tensor conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose2d: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose2d: weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose2d: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose2d: weight must be float32");
    TORCH_CHECK(input.dim() == 4, "conv_transpose2d: input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "conv_transpose2d: weight must be 4D (C_in,C_out,kH,kW)");
    TORCH_CHECK(stride > 0 && padding >= 0 && output_padding >= 0, "conv_transpose2d: invalid stride/padding/output_padding");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "conv_transpose2d: bias must be CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose2d: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "conv_transpose2d: bias must be 1D");
        bias_contig = bias.contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int N = input_contig.size(0);
    int C_in = input_contig.size(1);
    int H_in = input_contig.size(2);
    int W_in = input_contig.size(3);
    int C_out = weight_contig.size(1);
    int kH = weight_contig.size(2);
    int kW = weight_contig.size(3);

    int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;
    TORCH_CHECK(H_out > 0 && W_out > 0, "conv_transpose2d: output size too small");

    auto output = torch::empty({N, C_out, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose2d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool1d", &adaptive_avg_pool1d,
          "Custom adaptive average pooling 1D (CUDA)",
          pybind11::arg("input"), pybind11::arg("output_size"));
    m.def("conv_transpose2d", &conv_transpose2d,
          "Custom conv transpose 2D (CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
