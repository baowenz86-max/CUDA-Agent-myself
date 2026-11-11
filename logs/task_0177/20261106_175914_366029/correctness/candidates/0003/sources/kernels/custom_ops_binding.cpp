#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    bool has_bias,
    cudaStream_t stream
);

extern "C" void logsumexp_dim_forward_launcher(
    const float* input,
    float* output,
    int B,
    int C,
    int L,
    int input_stride_b,
    int input_stride_c,
    bool keepdim,
    int output_stride_b,
    int output_stride_c,
    cudaStream_t stream
);

// ConvTranspose1d implementation using custom CUDA kernel
static torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(1), "bias shape mismatch");
        bias_ptr = bias.data_ptr<float>();
    }

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(1);
    int64_t K = weight_contig.size(2);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(K),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        bias_ptr != nullptr,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after conv_transpose1d: ") + cudaGetErrorString(err));
    }

    return output;
}

// Logsumexp along dim (currently only supports dim == input.dim()-1)
static torch::Tensor logsumexp_dim_forward(
    const torch::Tensor& input,
    int64_t dim,
    bool keepdim
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(dim >= -input.dim() && dim < input.dim(), "dim out of range");

    int64_t dim_norm = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim_norm == input.dim() - 1, "only last dim logsumexp is implemented");
    TORCH_CHECK(input.dim() == 3, "expected 3D tensor [B, C, L] for logsumexp");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto input_contig = input.contiguous();

    std::vector<int64_t> output_sizes;
    if (keepdim) {
        output_sizes = {B, C, 1};
    } else {
        output_sizes = {B, C};
    }
    auto output = torch::empty(output_sizes, input_contig.options());

    int64_t output_stride_b, output_stride_c;
    if (keepdim) {
        output_stride_b = C * 1;
        output_stride_c = 1;
    } else {
        output_stride_b = C;
        output_stride_c = 1;
    }

    int64_t input_stride_b = C * L;
    int64_t input_stride_c = L;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    logsumexp_dim_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(input_stride_b),
        static_cast<int>(input_stride_c),
        keepdim,
        static_cast<int>(output_stride_b),
        static_cast<int>(output_stride_c),
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after logsumexp: ") + cudaGetErrorString(err));
    }

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", [](const torch::Tensor& input, const torch::Tensor& weight, const c10::optional<torch::Tensor>& bias, int64_t stride, int64_t padding, int64_t output_padding) {
        return conv_transpose1d_forward(input, weight, bias, stride, padding, output_padding);
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("output_padding") = 0);

    m.def("logsumexp_dim_forward", [](const torch::Tensor& input, int64_t dim, bool keepdim) {
        return logsumexp_dim_forward(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim"), py::arg("keepdim") = false);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
