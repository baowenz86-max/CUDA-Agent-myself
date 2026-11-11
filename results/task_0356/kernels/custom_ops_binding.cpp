#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void batch_norm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void group_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float eps,
    int N, int C, int H, int W, int num_groups,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float* mean_buffer,
    float* var_buffer,
    cudaStream_t stream);

// BatchNorm2d evaluation mode forward
torch::Tensor batch_norm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32 && weight.dim() == 1,
                "weight must be a contiguous CUDA float32 1D tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32 && bias.dim() == 1,
                "bias must be a contiguous CUDA float32 1D tensor");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.is_contiguous() && running_mean.scalar_type() == torch::kFloat32 && running_mean.dim() == 1,
                "running_mean must be a contiguous CUDA float32 1D tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.is_contiguous() && running_var.scalar_type() == torch::kFloat32 && running_var.dim() == 1,
                "running_var must be a contiguous CUDA float32 1D tensor");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK(weight.size(0) == C, "weight length must match C");
    TORCH_CHECK(bias.size(0) == C, "bias length must match C");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean length must match C");
    TORCH_CHECK(running_var.size(0) == C, "running_var length must match C");

    // Output tensor with same shape and strides as input
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());

    long long in_stride_n = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm_eval_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm_eval_launcher failed: ", cudaGetErrorString(err));
    return output;
}

// GroupNorm forward
torch::Tensor group_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    int64_t num_groups,
    int64_t num_channels) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32 && weight.dim() == 1,
                "weight must be a contiguous CUDA float32 1D tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32 && bias.dim() == 1,
                "bias must be a contiguous CUDA float32 1D tensor");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Set num_channels and num_groups defaults
    if (num_channels == -1) {
        num_channels = C;
    }
    TORCH_CHECK(num_channels == C, "num_channels must match input C");
    if (num_groups == -1) {
        num_groups = C;
    }
    TORCH_CHECK(num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    TORCH_CHECK(weight.size(0) == C, "weight length must match C");
    TORCH_CHECK(bias.size(0) == C, "bias length must match C");

    // Temporary buffers for per-(n, group) mean and variance
    auto mean_buffer = torch::empty({N * num_groups}, input.options());
    auto var_buffer = torch::empty({N * num_groups}, input.options());

    // Output tensor with same shape and strides as input
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());

    long long in_stride_n = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    group_norm_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<float>(eps),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(num_groups),
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        mean_buffer.data_ptr<float>(),
        var_buffer.data_ptr<float>(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_norm_launcher failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm2d evaluation mode forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05);
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-05,
          py::arg("num_groups") = -1,
          py::arg("num_channels") = -1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
