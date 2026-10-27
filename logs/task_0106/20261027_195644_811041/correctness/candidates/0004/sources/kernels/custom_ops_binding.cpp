#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void hardswish_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void conv_transpose1d_launcher(const float* input,
                                           const float* weight,
                                           const float* bias,
                                           float* output,
                                           int batch,
                                           int in_channels,
                                           int out_channels,
                                           int in_length,
                                           int out_length,
                                           int kernel_size,
                                           int stride,
                                           int padding,
                                           int output_padding,
                                           cudaStream_t stream);

#define CHECK_CUDA_ERR() { \
    cudaError_t err = cudaGetLastError(); \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
}

// Hardswish forward wrapper
torch::Tensor hardswish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardswish_forward_launcher(x_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    CHECK_CUDA_ERR();
    return output;
}

// Softplus forward wrapper
torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_forward_launcher(x_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    CHECK_CUDA_ERR();
    return output;
}

// ConvTranspose1d forward wrapper using custom CUDA kernel
torch::Tensor conv_transpose1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                      int64_t stride, int64_t padding, int64_t output_padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(stride > 0 && padding >= 0 && output_padding >= 0, "Invalid stride/padding/output_padding");

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int batch = x_contig.size(0);
    int in_channels = x_contig.size(1);
    int in_length = x_contig.size(2);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == in_channels, "Weight in_channels mismatch");
    TORCH_CHECK(bias_contig.numel() == out_channels, "Bias size mismatch");

    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "Invalid output length");

    auto output = torch::empty({batch, out_channels, out_length}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        (int)stride,
        (int)padding,
        (int)output_padding,
        stream
    );
    CHECK_CUDA_ERR();
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("hardswish_forward", &hardswish_forward, "Hardswish forward", pybind11::arg("x"));
    m.def("softplus_forward", &softplus_forward, "Softplus forward", pybind11::arg("x"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward, "ConvTranspose1d forward",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
