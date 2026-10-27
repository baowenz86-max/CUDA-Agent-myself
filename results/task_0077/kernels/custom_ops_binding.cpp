#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int L,
    float eps,
    cudaStream_t stream
);

void conv_transpose1d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int output_padding, int dilation, int groups,
    int L_out,
    cudaStream_t stream
);

// Python wrapper for conv_transpose1d_forward using custom CUDA kernel
// (cuDNN not available in this environment)
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions [N, C, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must have 3 dimensions [in_channels, out_channels/groups, kernel_size]");
    TORCH_CHECK(bias.dim() == 1, "bias must have 1 dimension [out_channels]");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    // Make contiguous if necessary (our kernel assumes contiguous memory)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(1) * groups;
    int64_t K = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == C_in, "weight shape mismatch: expected [", C_in, ", ", C_out / groups, ", ", K, "]");
    TORCH_CHECK(bias_contig.numel() == C_out, "bias size mismatch: expected ", C_out);

    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    TORCH_CHECK(L_out > 0, "computed output length is non-positive");

    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in,
        (int)C_out, (int)K,
        (int)stride, (int)padding, (int)output_padding, (int)dilation, (int)groups,
        (int)L_out,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv_transpose1d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Python wrapper for layer_norm_forward
torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions [N, C, L]");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions [C, L]");
    TORCH_CHECK(bias.dim() == 2, "bias must have 2 dimensions [C, L]");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    TORCH_CHECK(weight.size(0) == C && weight.size(1) == L, "weight shape must match [C, L]");
    TORCH_CHECK(bias.size(0) == C && bias.size(1) == L, "bias shape must match [C, L]");

    // Make contiguous if necessary
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    layer_norm_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        (int)N,
        (int)C,
        (int)L,
        (float)eps,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "layer_norm_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using custom CUDA kernel",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"),
          pybind11::arg("groups"),
          pybind11::arg("dilation"));

    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward over last two dimensions",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
