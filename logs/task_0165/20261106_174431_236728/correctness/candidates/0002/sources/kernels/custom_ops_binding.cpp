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

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding, bool has_bias,
    cudaStream_t stream);

extern "C" void conv_transpose2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in, int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding, bool has_bias,
    cudaStream_t stream);

namespace {

void check_cuda(cudaError_t status, const char* msg) {
    TORCH_CHECK(status == cudaSuccess, msg, cudaGetErrorString(status));
}

// Helper to compute output size
int64_t conv_transpose_out_size(int64_t in_size, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding) {
    return (in_size - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// ConvTranspose1d
torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight.size(0) must match input.size(1)");

    bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "bias.size(0) must match weight.size(1)");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);
    int64_t L_out = conv_transpose_out_size(L_in, K, stride, padding, 0);

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, has_bias,
        stream
    );

    check_cuda(cudaGetLastError(), "conv_transpose1d_launcher failed");
    return output;
}

// ConvTranspose2d
torch::Tensor conv_transpose2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight.size(0) must match input.size(1)");

    bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "bias.size(0) must match weight.size(1)");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t C_out = weight.size(1);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);
    int64_t H_out = conv_transpose_out_size(H_in, K_h, stride, padding, output_padding);
    int64_t W_out = conv_transpose_out_size(W_in, K_w, stride, padding, output_padding);

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        (int)N, (int)C_in, (int)H_in, (int)W_in, (int)C_out, (int)K_h, (int)K_w,
        (int)stride, (int)padding, (int)output_padding, has_bias,
        stream
    );

    check_cuda(cudaGetLastError(), "conv_transpose2d_launcher failed");
    return output;
}

// log_softmax binding
torch::Tensor log_softmax_forward(const torch::Tensor& input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");

    auto output = torch::empty_like(input);

    int ndims = input.dim();
    std::vector<int64_t> dims_vec(ndims);
    for (int i = 0; i < ndims; ++i) dims_vec[i] = input.size(i);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        dims_vec.data(),
        ndims,
        (int)dim,
        stream
    );

    check_cuda(cudaGetLastError(), "log_softmax_forward launcher failed");
    return output;
}

} // anonymous namespace

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward (custom CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "ConvTranspose2d forward (custom CUDA)",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "LogSoftmax forward",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);