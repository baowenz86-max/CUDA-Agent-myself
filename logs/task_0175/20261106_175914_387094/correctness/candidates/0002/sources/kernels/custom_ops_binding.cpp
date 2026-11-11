#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_group_norm_launcher(
    const float* x,
    float* y,
    int N,
    int C,
    int L,
    int groups,
    float eps,
    cudaStream_t stream
);

extern "C" void custom_conv_transpose1d_launcher(
    const float* x,
    const float* w,
    const float* bias,
    float* y,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    bool has_bias,
    cudaStream_t stream
);

// Helper to check CUDA errors
#define CHECK_CUDA_ERR() { \
    cudaError_t err = cudaGetLastError(); \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
}

// custom_group_norm binding
torch::Tensor custom_group_norm(torch::Tensor x, int64_t num_groups) {
    TORCH_CHECK(x.is_cuda(), "custom_group_norm: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_group_norm: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_group_norm: input must have 3 dimensions [batch, channels, length]");
    int64_t C = x.size(1);
    TORCH_CHECK(C % num_groups == 0, "custom_group_norm: channels must be divisible by num_groups");
    TORCH_CHECK(num_groups > 0, "custom_group_norm: num_groups must be positive");

    // Ensure input is contiguous; the kernel assumes contiguous memory
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    int N = static_cast<int>(x_contig.size(0));
    int C_int = static_cast<int>(x_contig.size(1));
    int L = static_cast<int>(x_contig.size(2));
    int groups = static_cast<int>(num_groups);
    float eps = 1e-5f;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_group_norm_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_int, L, groups, eps, stream
    );

    CHECK_CUDA_ERR();

    // If original tensor was non-contiguous, output is contiguous. Since model inputs are contiguous,
    // this is fine. For strict layout preservation, one would need to copy back, but not necessary here.
    return y;
}

// custom_conv_transpose1d binding using custom CUDA kernel (no cuDNN dependency)
torch::Tensor custom_conv_transpose1d(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(x.is_cuda(), "custom_conv_transpose1d: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "custom_conv_transpose1d: weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_conv_transpose1d: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "custom_conv_transpose1d: weight must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_conv_transpose1d: input must be 3D [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "custom_conv_transpose1d: weight must be 3D [in_channels, out_channels, kernel_size]");
    TORCH_CHECK(x.size(1) == weight.size(0), "custom_conv_transpose1d: input channels must match weight in_channels");
    TORCH_CHECK(stride > 0, "custom_conv_transpose1d: stride must be positive");
    TORCH_CHECK(padding >= 0, "custom_conv_transpose1d: padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "custom_conv_transpose1d: output_padding must be non-negative");

    bool has_bias = bias.has_value();
    const float* bias_ptr = nullptr;
    if (has_bias) {
        TORCH_CHECK(bias->is_cuda(), "custom_conv_transpose1d: bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "custom_conv_transpose1d: bias must be float32");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(1), "custom_conv_transpose1d: bias shape mismatch");
        TORCH_CHECK(bias->is_contiguous(), "custom_conv_transpose1d: bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    // Ensure input, weight contiguous
    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C_in = x_contig.size(1);
    int64_t L_in = x_contig.size(2);
    int64_t C_out = w_contig.size(1);
    int64_t K = w_contig.size(2);
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "custom_conv_transpose1d: computed output length is non-positive");

    auto y = torch::empty({N, C_out, L_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_conv_transpose1d_launcher(
        x_contig.data_ptr<float>(),
        w_contig.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(K),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        has_bias,
        stream
    );

    CHECK_CUDA_ERR();

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_group_norm", &custom_group_norm, "Group normalization custom op",
          pybind11::arg("x"), pybind11::arg("num_groups"));
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d, "Transposed 1D convolution custom op",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
