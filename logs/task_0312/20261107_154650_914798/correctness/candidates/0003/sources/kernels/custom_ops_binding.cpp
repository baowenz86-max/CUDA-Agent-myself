#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

namespace {

// Helper to get current CUDA stream
cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Declare launcher functions from .cu file
extern "C" void conv1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int dilation,
    int L_out,
    cudaStream_t stream);

extern "C" void maxpool1d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int L_in,
    int kernel_size, int stride, int padding, int dilation,
    int L_out,
    cudaStream_t stream);

// Conv1d forward
 torch::Tensor conv1d_cudnn_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "bias must be CUDA tensor or undefined");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_out, C_in, kernel)");

    // Make contiguous if necessary
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(1) == C_in, "weight input channels mismatch");
    if (bias_contig.defined()) {
        TORCH_CHECK(bias_contig.size(0) == C_out, "bias size mismatch");
    }

    // Compute output length
    int64_t L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = getCurrentStream();

    // Call launcher
    conv1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C_in), static_cast<int>(L_in),
        static_cast<int>(C_out), static_cast<int>(kernel_size),
        static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(dilation),
        static_cast<int>(L_out),
        stream
    );

    return output;
}

// MaxPool1d forward
 torch::Tensor maxpool1d_forward(
    const torch::Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");

    auto input_contig = input.contiguous();
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t L_in = input_contig.size(2);

    // Compute output length
    int64_t L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, L_out}, input_contig.options());

    cudaStream_t stream = getCurrentStream();

    maxpool1d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(L_in),
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(dilation),
        static_cast<int>(L_out),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv1d_cudnn_forward", &conv1d_cudnn_forward, "Conv1d forward using CUDA kernel",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias").none(true),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"));
    m.def("maxpool1d_forward", &maxpool1d_forward, "MaxPool1d forward using CUDA kernel",
          pybind11::arg("input"), pybind11::arg("kernel_size"), pybind11::arg("stride"),
          pybind11::arg("padding"), pybind11::arg("dilation"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
