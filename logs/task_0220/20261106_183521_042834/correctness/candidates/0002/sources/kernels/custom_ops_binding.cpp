#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_eval_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w,
    cudaStream_t stream
);

extern "C" void clamp_min_launcher(
    const float* input,
    float* output,
    float min_val,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w,
    cudaStream_t stream
);

// Helper to get strides from tensor (assuming 4D NCHW)
static void get_strides_4d(const torch::Tensor& t, int64_t& s_n, int64_t& s_c, int64_t& s_h, int64_t& s_w) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor");
    s_n = t.stride(0);
    s_c = t.stride(1);
    s_h = t.stride(2);
    s_w = t.stride(3);
}

torch::Tensor batch_norm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps = 1e-05
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    int64_t C = weight.size(0);
    TORCH_CHECK(input.size(1) == C, "input channel dim mismatch");

    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias shape mismatch");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == C, "running_mean shape mismatch");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == C, "running_var shape mismatch");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    int64_t N = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Allocate output with same shape and strides to preserve layout
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());

    // Get strides
    int64_t in_s_n, in_s_c, in_s_h, in_s_w;
    int64_t out_s_n, out_s_c, out_s_h, out_s_w;
    get_strides_4d(input, in_s_n, in_s_c, in_s_h, in_s_w);
    get_strides_4d(output, out_s_n, out_s_c, out_s_h, out_s_w);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    batch_norm_eval_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N, C, H, W,
        in_s_n, in_s_c, in_s_h, in_s_w,
        out_s_n, out_s_c, out_s_h, out_s_w,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm_eval_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor clamp_min_forward(torch::Tensor input, double min_val) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Allocate output with same shape and strides
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());

    int64_t in_s_n, in_s_c, in_s_h, in_s_w;
    int64_t out_s_n, out_s_c, out_s_h, out_s_w;
    get_strides_4d(input, in_s_n, in_s_c, in_s_h, in_s_w);
    get_strides_4d(output, out_s_n, out_s_c, out_s_h, out_s_w);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_min_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(min_val),
        N, C, H, W,
        in_s_n, in_s_c, in_s_h, in_s_w,
        out_s_n, out_s_c, out_s_h, out_s_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_min_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm2d evaluation forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps") = 1e-05);

    m.def("clamp_min_forward", &clamp_min_forward,
          "Clamp min forward",
          pybind11::arg("input"),
          pybind11::arg("min_val"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
