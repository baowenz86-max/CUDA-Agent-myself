#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mul_launcher(const float* x,
                                    const float* bias,
                                    float* output,
                                    int64_t batch_size,
                                    int64_t out_channels,
                                    int64_t length_out,
                                    cudaStream_t stream);

extern "C" void custom_conv_transpose1d_launcher(const float* input,
                                                 const float* weight,
                                                 const float* bias,
                                                 float* output,
                                                 int batch_size,
                                                 int in_channels,
                                                 int out_channels,
                                                 int length_in,
                                                 int length_out,
                                                 int kernel_size,
                                                 int stride,
                                                 int padding,
                                                 cudaStream_t stream);

// Helper to check CUDA errors
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(error));
    }
}

// Convert 1D ConvTranspose to custom kernel
// Input: x (N, C, L), weight (C, M, K), bias (M)
// Output: y (N, M, L_out)
// L_out = (L-1)*stride - 2*padding + K + output_padding
torch::Tensor custom_conv_transpose1d(torch::Tensor x,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      int64_t stride,
                                      int64_t padding,
                                      int64_t output_padding) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");

    int N = x.size(0);
    int C = x.size(1);
    int L = x.size(2);
    int K = weight.size(2);
    int M = weight.size(1);
    TORCH_CHECK(weight.size(0) == C, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == M, "bias size mismatch");

    // Ensure tensors are contiguous for kernel
    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    auto bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

    // Compute output length
    int64_t L_out = (L - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::zeros({N, M, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_conv_transpose1d_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, M, L, L_out, K, stride, padding,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("custom_conv_transpose1d kernel launch failed: ") + cudaGetErrorString(err));
    }

    return output;
}

// custom_mul: element-wise multiply with bias of shape (out_channels, 1)
torch::Tensor custom_mul(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D (out_channels, 1)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);
    TORCH_CHECK(bias.size(0) == C, "bias size[0] must equal channels");
    TORCH_CHECK(bias.size(1) == 1, "bias size[1] must be 1");

    auto output = torch::empty_like(x);

    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_mul_launcher(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, L,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("custom_mul kernel launch failed: ") + cudaGetErrorString(err));
    }

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
          "ConvTranspose1d using custom CUDA kernel",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("custom_mul", &custom_mul,
          "Element-wise multiply with bias broadcast",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
