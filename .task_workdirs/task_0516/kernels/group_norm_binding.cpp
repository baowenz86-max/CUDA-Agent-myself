#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_groups,
    const float* weight,
    const float* bias,
    double eps,
    int64_t input_stride_batch,
    int64_t input_stride_channel,
    int64_t input_stride_height,
    int64_t output_stride_batch,
    int64_t output_stride_channel,
    int64_t output_stride_height,
    cudaStream_t stream
);

torch::Tensor group_norm_forward(
    torch::Tensor input,
    int64_t num_groups,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    double eps) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D: [batch, channels, height, width]");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    int64_t channels = input.size(1);
    TORCH_CHECK(channels % num_groups == 0, "channels must be divisible by num_groups");

    // Validate optional weight and bias
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight->numel() == channels, "weight must have exactly channels elements");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->numel() == channels, "bias must have exactly channels elements");
        bias_ptr = bias->data_ptr<float>();
    }

    // Output: same shape as input. Use empty_like to preserve strides/layout, but ensure contiguous data? 
    // The launcher writes output using output strides; if input is non-contiguous, output should match input stride to satisfy contract.
    // However, cuda kernel writes with provided strides; output tensor can be non-contiguous matching input strides.
    auto output = torch::empty_like(input);

    int64_t batch_size = input.size(0);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    int64_t input_stride_batch = input.stride(0);
    int64_t input_stride_channel = input.stride(1);
    int64_t input_stride_height = input.stride(2);
    int64_t output_stride_batch = output.stride(0);
    int64_t output_stride_channel = output.stride(1);
    int64_t output_stride_height = output.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        num_groups,
        weight_ptr,
        bias_ptr,
        eps,
        input_stride_batch,
        input_stride_channel,
        input_stride_height,
        output_stride_batch,
        output_stride_channel,
        output_stride_height,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_norm kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "Group normalization forward",
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"));
}

REGISTER_BINDING(group_norm, register_group_norm);
