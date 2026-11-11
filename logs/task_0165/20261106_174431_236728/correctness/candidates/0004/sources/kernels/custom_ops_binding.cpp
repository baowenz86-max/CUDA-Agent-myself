#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_forward_launcher(
    float* output,
    const float* input,
    const int64_t* dims,
    int ndims,
    int dim,
    cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding,
    int L_out,
    bool has_bias,
    cudaStream_t stream);

extern "C" void conv_transpose2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding,
    int H_out, int W_out,
    bool has_bias,
    cudaStream_t stream);

namespace {

void check_cuda(cudaError_t status, const char* msg) {
    TORCH_CHECK(status == cudaSuccess, msg, cudaGetErrorString(status));
}

// ConvTranspose1d forward
// Parameters:
// - input: contiguous float32 CUDA tensor [N, C_in, L_in]
// - weight: contiguous float32 CUDA tensor [C_in, C_out, K]
// - bias: optional contiguous float32 CUDA tensor [C_out]
// - stride: int
// - padding: int
// Returns output tensor [N, C_out, L_out]
torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight.size(0) must match input.size(1)");

    const float* bias_ptr = nullptr;
    bool has_bias = false;
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "bias size must match out_channels");
        bias_ptr = b.data_ptr<float>();
        has_bias = true;
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1);
    int K = weight.size(2);
    int output_padding = 0;
    int L_out = (L_in - 1) * (int)stride - 2 * (int)padding + K + output_padding;

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, L_out,
        has_bias,
        stream);

    check_cuda(cudaGetLastError(), "conv_transpose1d_forward launcher failed");
    return output;
}

// ConvTranspose2d forward
// input: [N,C_in,H,W], weight: [C_in,C_out,Kh,Kw]
// stride, padding, output_padding scalar ints
// output: [N,C_out,H_out,W_out]
torch::Tensor conv_transpose2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "conv_transpose2d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose2d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight.size(0) must match input.size(1)");

    const float* bias_ptr = nullptr;
    bool has_bias = false;
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "bias size must match out_channels");
        bias_ptr = b.data_ptr<float>();
        has_bias = true;
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    int H_out = (H_in - 1) * (int)stride - 2 * (int)padding + K_h + (int)output_padding;
    int W_out = (W_in - 1) * (int)stride - 2 * (int)padding + K_w + (int)output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose2d_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        N, C_in, H_in, W_in, C_out, K_h, K_w,
        (int)stride, (int)padding, (int)output_padding,
        H_out, W_out,
        has_bias,
        stream);

    check_cuda(cudaGetLastError(), "conv_transpose2d_forward launcher failed");
    return output;
}

// log_softmax binding
torch::Tensor log_softmax_forward(const torch::Tensor& input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "log_softmax_forward: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");

    auto output = torch::empty_like(input);

    int ndims = input.dim();
    std::vector<int64_t> dims_vec(ndims);
    for (int i = 0; i < ndims; ++i) {
        dims_vec[i] = input.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        dims_vec.data(),
        ndims,
        (int)dim,
        stream);

    check_cuda(cudaGetLastError(), "log_softmax_forward launcher failed");
    return output;
}

} // namespace

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "ConvTranspose2d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "LogSoftmax forward",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
