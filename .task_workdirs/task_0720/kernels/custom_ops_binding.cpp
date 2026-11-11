// custom_ops_binding.cpp
// PyTorch bindings for custom_ops.cu (linear_forward)
// No CUDA kernel implementation; only tensor validation, allocation, and launcher call.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
// All pointers are raw device pointers; dimensions and strides are int64_t.
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t B, int64_t L, int64_t K, int64_t N,
    int64_t input_stride_b, int64_t input_stride_l, int64_t input_stride_k,
    int64_t weight_stride_n, int64_t weight_stride_k,
    int64_t output_stride_b, int64_t output_stride_l, int64_t output_stride_n,
    cudaStream_t stream);

// Wrapper function matching Python signature: linear_forward(input, weight, bias)
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // 1. Device checks
    TORCH_CHECK(input.is_cuda(), "linear_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "linear_forward: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "linear_forward: bias must be a CUDA tensor");

    // 2. Dtype checks (contract specifies float32 only)
    TORCH_CHECK(input.dtype() == torch::kFloat32, "linear_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear_forward: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "linear_forward: bias must be float32");

    // 3. Shape checks
    // input: [B, L, K]
    // weight: [N, K]
    // bias: [N]
    TORCH_CHECK(input.dim() == 3, "linear_forward: input must be 3D (B, L, K)");
    TORCH_CHECK(weight.dim() == 2, "linear_forward: weight must be 2D (N, K)");
    TORCH_CHECK(bias.dim() == 1, "linear_forward: bias must be 1D (N)");

    int64_t B = input.size(0);
    int64_t L = input.size(1);
    int64_t K = input.size(2);
    int64_t N = weight.size(0);
    int64_t K_weight = weight.size(1);
    int64_t N_bias = bias.size(0);

    TORCH_CHECK(K == K_weight, "linear_forward: input feature dim must match weight feature dim");
    TORCH_CHECK(N == N_bias, "linear_forward: weight output dim must match bias size");

    // 4. Make contiguous if necessary (launcher assumes row-major with provided strides)
    // The launcher handles non-contiguous strides, but contiguous improves performance
    // and simplifies potential bugs. We obtain contiguous versions.
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // 5. Allocate output tensor [B, L, N] on same device, float32
    auto output = torch::empty({B, L, N}, input.options());

    // 6. Compute strides (in elements)
    int64_t input_stride_b = input_contig.stride(0);
    int64_t input_stride_l = input_contig.stride(1);
    int64_t input_stride_k = input_contig.stride(2);
    int64_t weight_stride_n = weight_contig.stride(0);
    int64_t weight_stride_k = weight_contig.stride(1);
    int64_t output_stride_b = output.stride(0);
    int64_t output_stride_l = output.stride(1);
    int64_t output_stride_n = output.stride(2);

    // 7. Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 8. Call launcher with raw device pointers (types match launcher: const float*)
    linear_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L, K, N,
        input_stride_b, input_stride_l, input_stride_k,
        weight_stride_n, weight_stride_k,
        output_stride_b, output_stride_l, output_stride_n,
        stream
    );

    // 9. Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_linear_forward(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear transformation: output = input @ weight^T + bias",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"));
}

// Auto-register binding
REGISTER_BINDING(linear_forward, register_linear_forward);
