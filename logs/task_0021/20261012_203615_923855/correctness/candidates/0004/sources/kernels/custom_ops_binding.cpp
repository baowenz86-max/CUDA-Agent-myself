#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias,
    float* output, int B, int C_in, int L, int C_out, int K,
    cudaStream_t stream);

extern "C" void batch_matmul_launcher(
    const float* A, float* out,
    int B, int M, int N,
    cudaStream_t stream);

extern "C" void softplus_launcher(
    const float* input, float* output, int count,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    const float* input, const float* bias, float* output,
    int B, int C, int N,
    cudaStream_t stream);

// Wrapper: conv1d_forward(input, weight, bias) -> Tensor
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C_in, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_out, C_in, K]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C_out]");

    int B = input.size(0);
    int C_in = input.size(1);
    int L = input.size(2);
    int C_out = weight.size(0);
    int K = weight.size(2);
    TORCH_CHECK(weight.size(1) == C_in, "weight C_in mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias size mismatch");

    auto output = torch::empty({B, C_out, L - K + 1}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, L, C_out, K,
        stream);
    return output;
}

// Wrapper: batch_matmul_forward(input) -> Tensor, computes input @ input^T
torch::Tensor batch_matmul_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, M, N]");

    int B = input.size(0);
    int M = input.size(1);
    int N = input.size(2);
    auto output = torch::empty({B, M, M}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_matmul_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, M, N,
        stream);
    return output;
}

// Wrapper: softplus_forward(input) -> Tensor
torch::Tensor softplus_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int count = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        count,
        stream);
    return output;
}

// Wrapper: add_bias_forward(input, bias) -> Tensor (bias shape [C] or [C,1])
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C, N]");
    TORCH_CHECK(bias.numel() == input.size(1), "bias must have numel == C");

    int B = input.size(0);
    int C = input.size(1);
    int N = input.size(2);
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, N,
        stream);
    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("conv1d_forward", &conv1d_forward, "Conv1d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("batch_matmul_forward", &batch_matmul_forward, "Batch matmul A @ A^T",
          pybind11::arg("input"));
    m.def("softplus_forward", &softplus_forward, "Softplus",
          pybind11::arg("input"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias",
          pybind11::arg("input"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
