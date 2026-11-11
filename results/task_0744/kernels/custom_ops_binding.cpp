// kernels/custom_ops_binding.cpp
// PyTorch bindings for mean_dim0_keepdim and batchnorm2d_eval.
// No torch:: operators used for computation; only tensor checks, allocation,
// stream retrieval, and launcher calls.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" {
void mean_dim0_keepdim_cuda(
    const float* input_ptr,
    float* output_ptr,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t input_stride_batch,
    int64_t input_stride_channel,
    int64_t input_stride_height,
    int64_t input_stride_width,
    cudaStream_t stream);

void batchnorm2d_eval_cuda(
    const float* input_ptr,
    float* output_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    const float* running_mean_ptr,
    const float* running_var_ptr,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_N,
    int64_t input_stride_C,
    int64_t input_stride_H,
    int64_t input_stride_W,
    int64_t output_stride_N,
    int64_t output_stride_C,
    int64_t output_stride_H,
    int64_t output_stride_W,
    cudaStream_t stream);
}

// Wrapper for mean_dim0_keepdim
torch::Tensor mean_dim0_keepdim(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "mean_dim0_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim0_keepdim: input must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_dim0_keepdim: input must be 4D");
    // No contiguity requirement due to support of strides, but we pass actual strides.

    auto batch = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    // Output shape: [1, channels, height, width]
    auto output = torch::empty({1, channels, height, width}, x.options());

    // Obtain current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mean_dim0_keepdim_cuda(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(channels),
        static_cast<int64_t>(height),
        static_cast<int64_t>(width),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream
    );

    // Check for launch errors (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim0_keepdim: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for batchnorm2d_eval
torch::Tensor batchnorm2d_eval(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Validate input
    TORCH_CHECK(input.is_cuda(), "batchnorm2d_eval: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "batchnorm2d_eval: input must be float32");
    TORCH_CHECK(input.dim() == 4, "batchnorm2d_eval: input must be 4D");

    // Validate running stats
    TORCH_CHECK(running_mean.is_cuda(), "batchnorm2d_eval: running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "batchnorm2d_eval: running_var must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "batchnorm2d_eval: running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "batchnorm2d_eval: running_var must be float32");
    TORCH_CHECK(running_mean.dim() == 1 && running_var.dim() == 1, "batchnorm2d_eval: running stats must be 1D");
    TORCH_CHECK(running_mean.numel() == input.size(1), "batchnorm2d_eval: running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == input.size(1), "batchnorm2d_eval: running_var size mismatch");

    // Optionally validate weight and bias
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "batchnorm2d_eval: weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "batchnorm2d_eval: weight must be float32");
        TORCH_CHECK(w.dim() == 1 && w.numel() == input.size(1), "batchnorm2d_eval: weight size mismatch");
        weight_ptr = w.data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "batchnorm2d_eval: bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "batchnorm2d_eval: bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.numel() == input.size(1), "batchnorm2d_eval: bias size mismatch");
        bias_ptr = b.data_ptr<float>();
    }

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batchnorm2d_eval_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        C,
        H,
        W,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batchnorm2d_eval: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim0_keepdim", &mean_dim0_keepdim,
          "Compute mean over dimension 0 with keepdim=True",
          py::arg("x"));

    m.def("batchnorm2d_eval", &batchnorm2d_eval,
          "Apply BatchNorm2d evaluation using running statistics",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
