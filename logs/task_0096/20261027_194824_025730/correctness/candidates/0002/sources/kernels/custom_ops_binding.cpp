#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_forward_float32(
    const float* input,
    float* output,
    int B, int C, int L,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_l,
    cudaStream_t stream
);

extern "C" void conv_transpose1d_forward_float32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream
);

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (C_out)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    int64_t B = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "computed output length is non-positive");

    // Ensure contiguity (parameters are already contiguous, but input may not be)
    auto input_contig = input.is_contiguous() ? input : input.contiguous();
    auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    auto bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

    // Allocate output
    auto output = torch::empty({B, C_out, L_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with int cast (safe since sizes are within int range for this task)
    conv_transpose1d_forward_float32(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(K),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    return output;
}

torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");
    TORCH_CHECK(input.dim() == 3, "softmax currently only supports 3D tensors");
    TORCH_CHECK(dim == 1, "softmax currently only supports dim=1");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long stride_b = input.stride(0);
    long long stride_c = input.stride(1);
    long long stride_l = input.stride(2);

    softmax_forward_float32(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        stride_b,
        stride_c,
        stride_l,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d", &conv_transpose1d_forward,
          "1D transposed convolution (custom CUDA kernel)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("output_padding") = 0);
    m.def("softmax", &softmax_forward,
          "Softmax along a dimension",
          pybind11::arg("input"),
          pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
