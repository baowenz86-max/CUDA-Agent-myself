// custom_ops_binding.cpp : PyTorch bindings for custom_ops.cu
// All Tensor checks, allocations, and pybind registrations.
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in, int C_out, int K, int L_out,
    cudaStream_t stream
);

extern "C" void batch_matmul_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    cudaStream_t stream
);

extern "C" void softplus_launcher(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
);

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int N,
    cudaStream_t stream
);

// Helper to check CUDA float32 contiguous
static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// conv1d_forward: input [B, C_in, L_in], weight [C_out, C_in, K], bias [C_out] optional
// outputs [B, C_out, L_out]
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(weight, "weight");
    if (bias.defined()) {
        check_cuda_float32_contiguous(bias, "bias");
    }

    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C_in, L_in]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_out, C_in, K]");

    int B = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(0);
    int K = weight.size(2);
    TORCH_CHECK(weight.size(1) == C_in, "weight input channels must match");
    TORCH_CHECK(weight.size(2) == K, "kernel size mismatch");
    TORCH_CHECK(K == 3, "only kernel size 3 supported"); // hardcoded for model
    int L_out = L_in - K + 1;
    TORCH_CHECK(L_out > 0, "input length too small for kernel");
    if (bias.defined()) {
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "bias must be 1D [C_out]");
    }

    auto output = torch::empty({B, C_out, L_out}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, L_in, C_out, K, L_out,
        stream
    );
    return output;
}

// batch_matmul_forward: input [B, C, L] -> output [B, C, C] = A @ A^T
torch::Tensor batch_matmul_forward(torch::Tensor input) {
    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C, L]");
    int B = input.size(0);
    int C = input.size(1);
    int L = input.size(2);
    auto output = torch::empty({B, C, C}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_matmul_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, L,
        stream
    );
    return output;
}

// softplus_forward: elementwise softplus
torch::Tensor softplus_forward(torch::Tensor input) {
    check_cuda_float32_contiguous(input, "input");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );
    return output;
}

// add_bias_forward: input [B, C, N], bias [C] or [C,1] -> output [B, C, N]
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C, N]");
    TORCH_CHECK(bias.dim() == 1 || (bias.dim() == 2 && bias.size(1) == 1), "bias must be 1D [C] or 2D [C,1]");
    int B = input.size(0);
    int C = input.size(1);
    int N = input.size(2);
    TORCH_CHECK(bias.numel() == C, "bias numel must match C dimension");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, N,
        stream
    );
    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("conv1d_forward", &conv1d_forward, "Conv1d forward (custom CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("batch_matmul_forward", &batch_matmul_forward, "Batch matmul A @ A^T",
          pybind11::arg("input"));
    m.def("softplus_forward", &softplus_forward, "Softplus forward",
          pybind11::arg("input"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias broadcast",
          pybind11::arg("input"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
