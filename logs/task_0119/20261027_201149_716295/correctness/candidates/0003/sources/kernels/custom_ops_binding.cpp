#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm2d_inference_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    const long long total_elements,
    const int N,
    const int C,
    const int H,
    const int W,
    const long long input_N_stride,
    const long long input_C_stride,
    const long long input_H_stride,
    const long long input_W_stride,
    const long long output_N_stride,
    const long long output_C_stride,
    const long long output_H_stride,
    const long long output_W_stride,
    const float eps,
    cudaStream_t stream);

extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    const long long total_elements,
    const float negative_slope,
    cudaStream_t stream);

// PyTorch wrapper for batch_norm2d_inference
torch::Tensor batch_norm2d_inference(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32,
                "running_mean must be CUDA float32");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32,
                "running_var must be CUDA float32");
    TORCH_CHECK(running_mean.dim() == 1 && running_var.dim() == 1,
                "running_mean/var must be 1D");
    TORCH_CHECK(running_mean.size(0) == input.size(1),
                "running_mean size must match channel dimension");
    TORCH_CHECK(running_var.size(0) == input.size(1),
                "running_var size must match channel dimension");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32,
                    "weight must be CUDA float32");
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1),
                    "weight size must match channel dimension");
        weight = weight.contiguous();
        weight_ptr = weight.data_ptr<float>();
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32,
                    "bias must be CUDA float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == input.size(1),
                    "bias size must match channel dimension");
        bias = bias.contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    // Make input contiguous (simplifies stride handling, but safe)
    input = input.contiguous();
    running_mean = running_mean.contiguous();
    running_var = running_var.contiguous();

    auto output = torch::empty_like(input);

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    long long total_elements = input.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides after contiguous: N*C*H*W, C*H*W, H*W, W, 1 (in elements)
    long long input_N_stride = C * H * W;
    long long input_C_stride = H * W;
    long long input_H_stride = W;
    long long input_W_stride = 1;
    long long output_N_stride = input_N_stride;
    long long output_C_stride = input_C_stride;
    long long output_H_stride = input_H_stride;
    long long output_W_stride = input_W_stride;

    batch_norm2d_inference_launcher(
        input.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        (int)N,
        (int)C,
        (int)H,
        (int)W,
        input_N_stride,
        input_C_stride,
        input_H_stride,
        input_W_stride,
        output_N_stride,
        output_C_stride,
        output_H_stride,
        output_W_stride,
        eps,
        stream
    );

    return output;
}

// PyTorch wrapper for leaky_relu_forward
torch::Tensor leaky_relu_forward(
    torch::Tensor input,
    float negative_slope)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    input = input.contiguous();
    auto output = torch::empty_like(input);

    long long total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        negative_slope,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_inference", &batch_norm2d_inference,
          "BatchNorm2d inference forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("input"),
          py::arg("negative_slope"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
