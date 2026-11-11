#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void leaky_relu_forward_launcher(
    float* output,
    const float* input,
    long long numel,
    float negative_slope,
    cudaStream_t stream
);

extern "C" void instance_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    float eps,
    cudaStream_t stream
);

// LeakyReLU wrapper
static torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope = 0.01) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.numel() > 0, "Input must have at least one element");

    // Preserve original stride/size for output
    auto original_sizes = input.sizes();
    auto original_strides = input.strides();

    // Ensure contiguous for simple elementwise launcher
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        static_cast<float>(negative_slope),
        stream
    );

    // Check for kernel errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "leaky_relu_forward kernel launch failed");

    // Restore original stride by as_strided if needed
    if (!input.is_contiguous()) {
        return output.as_strided(original_sizes, original_strides);
    }
    return output;
}

// InstanceNorm wrapper
static torch::Tensor instance_norm_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "InstanceNorm2d expects 4D input (N,C,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Weight and bias checks
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight.value().defined()) {
        TORCH_CHECK(weight.value().is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.value().dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.value().numel() == C, "weight shape must match C");
        weight_ptr = weight.value().contiguous().data_ptr<float>();
    }
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.value().numel() == C, "bias shape must match C");
        bias_ptr = bias.value().contiguous().data_ptr<float>();
    }

    // Ensure input is contiguous? launcher supports strides but we can also pass original
    // If input is non-contiguous, the launcher can still handle strides, so no need to contiguous.
    // However we must ensure the output has same shape and stride.
    auto output = torch::empty_like(input);

    // Strides (in elements)
    long long n_stride = input.stride(0);
    long long c_stride = input.stride(1);
    long long h_stride = input.stride(2);
    long long w_stride = input.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    instance_norm_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        N, C, H, W,
        n_stride, c_stride, h_stride, w_stride,
        static_cast<float>(eps),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "instance_norm_forward kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("input"),
          py::arg("negative_slope") = 0.01);
    m.def("instance_norm_forward", &instance_norm_forward,
          "InstanceNorm2d forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
