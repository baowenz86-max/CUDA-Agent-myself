#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int L_in, int L_out, int K,
    cudaStream_t stream);

extern "C" void batch_matmul_launcher(
    const float* A,
    float* output,
    int B, int N, int K,
    cudaStream_t stream);

extern "C" void softplus_launcher(
    const float* input,
    float* output,
    int total,
    float threshold,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C_out, int N,
    cudaStream_t stream);

// Conv1d wrapper
at::Tensor conv1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");

    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C_in, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_out, C_in, K]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C_out]");

    int B = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(0);
    int K = weight.size(2);

    TORCH_CHECK(weight.size(1) == C_in, "weight C_in dimension mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias length must match C_out");

    int L_out = L_in - K + 1;

    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();

    auto output = at::empty({B, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv1d_launcher(
        input_c.data_ptr<float>(),
        weight_c.data_ptr<float>(),
        bias_c.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, L_in, L_out, K,
        stream);

    return output;
}

// Batch Matmul wrapper: A is [B, N, K], output is [B, N, N]
at::Tensor batch_matmul_forward(at::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == at::kFloat, "A must be float32");
    TORCH_CHECK(A.dim() == 3, "A must be 3D [B, N, K]");

    int B = A.size(0);
    int N = A.size(1);
    int K = A.size(2);

    auto A_c = A.contiguous();
    auto output = at::empty({B, N, N}, A.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_matmul_launcher(
        A_c.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, K,
        stream);

    return output;
}

// Softplus wrapper (elementwise)
at::Tensor softplus_forward(at::Tensor input, double threshold = 20.0) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");

    auto input_c = input.contiguous();
    auto output = at::empty_like(input_c);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(
        input_c.data_ptr<float>(),
        output.data_ptr<float>(),
        input_c.numel(),
        static_cast<float>(threshold),
        stream);

    return output;
}

// Add bias wrapper: input [B, C, N], bias [C] or [C,1]
at::Tensor add_bias_forward(at::Tensor input, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, C, N]");
    TORCH_CHECK(bias.dim() == 1 || bias.dim() == 2, "bias must be 1D or 2D");

    int B = input.size(0);
    int C = input.size(1);
    int N = input.size(2);

    int C_bias = (bias.dim() == 1) ? bias.size(0) : bias.size(0);
    TORCH_CHECK(C_bias == C, "bias length must match C");
    if (bias.dim() == 2) {
        TORCH_CHECK(bias.size(1) == 1, "bias second dim must be 1");
    }

    auto input_c = input.contiguous();
    auto bias_c = bias.contiguous();
    auto output = at::empty_like(input_c);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        input_c.data_ptr<float>(),
        bias_c.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, N,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv1d_forward", &conv1d_forward, "Conv1d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("batch_matmul_forward", &batch_matmul_forward, "Batch matmul forward",
          pybind11::arg("A"));
    m.def("softplus_forward", &softplus_forward, "Softplus forward",
          pybind11::arg("input"), pybind11::arg("threshold") = 20.0);
    m.def("add_bias_forward", &add_bias_forward, "Add bias forward",
          pybind11::arg("input"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);